from __future__ import annotations

from .audit_log import append_audit_event, build_audit_event

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

POLICY_VERSION = "AUTO_MODEL_POLICY_V1.0"

BucketName = Literal["FAST", "REASONING", "SAFE_SMALL"]


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class AutoModelInputs:
    actor: str
    endpoint: Literal["chat", "route"]
    requested_model: str
    max_tokens: int
    temperature: Optional[float]
    message_count: int
    has_system_prompt: bool
    risk_score: Optional[float]
    allow_remote_models: bool = False
    task_type: Optional[str] = None
    messages: Optional[List[ChatMessage]] = None


@dataclass(frozen=True)
class AutoModelDecision:
    chosen_model: str
    reason: str
    escalation: bool
    applied_constraints: List[str] = field(default_factory=list)
    tools_disabled: bool = False

    policy_version: str = POLICY_VERSION
    decision_hash: str = ""


@dataclass(frozen=True)
class ActorPolicy:
    actor: str
    allowed_models: frozenset[str]
    denied_models: frozenset[str] = frozenset()
    allow_remote: bool = False
    force_bucket: Optional[BucketName] = None
    tools_disabled: bool = False

    def allows_model(self, model_name: str, is_remote: bool) -> bool:
        if model_name in self.denied_models:
            return False
        if self.allowed_models and model_name not in self.allowed_models:
            return False
        if is_remote and not self.allow_remote:
            return False
        return True


@dataclass(frozen=True)
class ModelCatalog:
    bucket_models: Dict[BucketName, List[str]]
    available_models: frozenset[str]
    remote_models: frozenset[str] = frozenset()

    def is_available(self, model_name: str) -> bool:
        return model_name in self.available_models

    def is_remote(self, model_name: str) -> bool:
        return model_name in self.remote_models

    def first_available_allowed(self, bucket: BucketName, policy: ActorPolicy) -> Optional[str]:
        for model in self.bucket_models.get(bucket, []):
            if self.is_available(model) and policy.allows_model(model, self.is_remote(model)):
                return model
        return None


def _normalize_actor_policy(actor: str) -> ActorPolicy:
    actor_lc = (actor or "").strip().lower()

    if actor_lc == "public":
        return ActorPolicy(
            actor="public",
            allowed_models=frozenset({"safe_primary", "safe_secondary"}),
            allow_remote=False,
            force_bucket="SAFE_SMALL",
            tools_disabled=True,
        )

    if actor_lc == "rainbow":
        return ActorPolicy(
            actor="rainbow",
            allowed_models=frozenset(
                {
                    "fast_primary",
                    "fast_secondary",
                    "last_known_good_fast",
                    "reasoning_primary",
                    "reasoning_secondary",
                    "last_known_good_reasoning",
                    "safe_primary",
                    "safe_secondary",
                }
            ),
            allow_remote=False,
        )

    # default conservative internal policy
    return ActorPolicy(
        actor=actor_lc or "unknown",
        allowed_models=frozenset(
            {
                "fast_primary",
                "fast_secondary",
                "last_known_good_fast",
                "safe_primary",
                "safe_secondary",
            }
        ),
        allow_remote=False,
    )


def build_actor_policy(actor: str, allow_remote_models: bool = False) -> ActorPolicy:
    base = _normalize_actor_policy(actor)
    if not allow_remote_models:
        return base
    return ActorPolicy(
        actor=base.actor,
        allowed_models=base.allowed_models,
        denied_models=base.denied_models,
        allow_remote=base.allow_remote and allow_remote_models,
        force_bucket=base.force_bucket,
        tools_disabled=base.tools_disabled,
    )


def derive_task_type(messages: Optional[Iterable[ChatMessage]]) -> str:
    if not messages:
        return "general"
    text = " ".join((m.content or "").lower() for m in messages)
    if any(k in text for k in ("patch", "diff", "refactor", "function", "bug", "test")):
        return "coding"
    if any(k in text for k in ("plan", "architecture", "reason", "design", "analyze")):
        return "reasoning"
    if any(k in text for k in ("image", "screenshot", "vision")):
        return "vision"
    return "general"


def _preferred_bucket(inputs: AutoModelInputs, policy: ActorPolicy) -> BucketName:
    if policy.force_bucket:
        return policy.force_bucket
    if inputs.max_tokens >= 300:
        return "REASONING"
    return "FAST"


def _fallback_model(bucket: BucketName, catalog: ModelCatalog, policy: ActorPolicy) -> Optional[str]:
    chain: List[BucketName] = [bucket]
    if bucket != "FAST":
        chain.append("FAST")
    if "SAFE_SMALL" not in chain:
        chain.append("SAFE_SMALL")

    for b in chain:
        chosen = catalog.first_available_allowed(b, policy)
        if chosen:
            return chosen
    return None


def _decision_hash(inputs: AutoModelInputs, decision: AutoModelDecision) -> str:
    constraints_sorted = sorted(list(decision.applied_constraints or []))

    payload = {
        "policy_version": POLICY_VERSION,
        "actor": inputs.actor,
        "endpoint": inputs.endpoint,
        "requested_model": inputs.requested_model,
        "max_tokens": inputs.max_tokens,
        "temperature": inputs.temperature,
        "message_count": inputs.message_count,
        "has_system_prompt": inputs.has_system_prompt,
        "risk_score": inputs.risk_score,
        "allow_remote_models": inputs.allow_remote_models,
        "task_type": inputs.task_type,
        # decision outputs
        "chosen_model": decision.chosen_model,
        "reason": decision.reason,
        "escalation": decision.escalation,
        "tools_disabled": decision.tools_disabled,
        "applied_constraints": constraints_sorted,
    }

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _finalize_decision(inputs: AutoModelInputs, d: AutoModelDecision) -> AutoModelDecision:
    # NOTE: hash computed on canonical decision + sorted constraints inside _decision_hash
    return AutoModelDecision(
        chosen_model=d.chosen_model,
        reason=d.reason,
        escalation=d.escalation,
        applied_constraints=list(d.applied_constraints or []),
        tools_disabled=d.tools_disabled,
        policy_version=POLICY_VERSION,
        decision_hash=_decision_hash(inputs, d),
    )

def _emit_audit(inputs: AutoModelInputs, decision: AutoModelDecision) -> None:
    ev = build_audit_event(
        policy_version=decision.policy_version,
        actor=inputs.actor,
        endpoint=inputs.endpoint,
        requested_model=inputs.requested_model,
        chosen_model=decision.chosen_model,
        reason=decision.reason,
        escalation=decision.escalation,
        tools_disabled=decision.tools_disabled,
        decision_hash=decision.decision_hash,
        max_tokens=inputs.max_tokens,
        temperature=inputs.temperature,
        message_count=inputs.message_count,
        has_system_prompt=inputs.has_system_prompt,
        risk_score=inputs.risk_score,
        task_type=inputs.task_type,
    )
    append_audit_event(ev)


def resolve_auto_model(inputs: AutoModelInputs, catalog: ModelCatalog, policy: ActorPolicy) -> AutoModelDecision:
    constraints: List[str] = []

    if policy.tools_disabled:
        constraints.append("TOOLS_DISABLED")
    if not inputs.allow_remote_models:
        constraints.append("REMOTE_REQUEST_DISABLED")
    if not policy.allow_remote:
        constraints.append("REMOTE_POLICY_DISABLED")

    preferred_bucket = _preferred_bucket(inputs, policy)

    requested = (inputs.requested_model or "").strip() or "auto"
    if requested != "auto":
        requested_remote = catalog.is_remote(requested)

        # Requested model is forbidden by policy -> downgrade to fallback (escalation)
        if not policy.allows_model(requested, requested_remote):
            fallback = _fallback_model(preferred_bucket, catalog, policy)
            if not fallback:
                raise ValueError("NO_ALLOWED_MODEL_AVAILABLE")

            d = AutoModelDecision(
                chosen_model=fallback,
                reason="DOWNGRADE_FORBIDDEN",
                escalation=True,
                applied_constraints=constraints,
                tools_disabled=policy.tools_disabled,
            )
            out = _finalize_decision(inputs, d)
            _emit_audit(inputs, out)
            return out

        # Requested model is allowed and available -> use it (no escalation)
        if catalog.is_available(requested):
            reason = "REMOTE_ALLOWED" if requested_remote else "REQUESTED_ALLOWED"
            d = AutoModelDecision(
                chosen_model=requested,
                reason=reason,
                escalation=False,
                applied_constraints=constraints,
                tools_disabled=policy.tools_disabled,
            )
            out = _finalize_decision(inputs, d)
            _emit_audit(inputs, out)
            return out

        # Requested model is allowed but unavailable -> fallback (escalation)
        fallback = _fallback_model(preferred_bucket, catalog, policy)
        if not fallback:
            raise ValueError("NO_ALLOWED_MODEL_AVAILABLE")

        d = AutoModelDecision(
            chosen_model=fallback,
            reason="FALLBACK_UNAVAILABLE",
            escalation=True,
            applied_constraints=constraints,
            tools_disabled=policy.tools_disabled,
        )
        out = _finalize_decision(inputs, d)
        _emit_audit(inputs, out)
        return out

    # AUTO selection path (no explicit request)
    selected = catalog.first_available_allowed(preferred_bucket, policy)
    if selected:
        reason = "PUBLIC_SAFE" if policy.force_bucket == "SAFE_SMALL" else (
            "AUTO_REASONING" if preferred_bucket == "REASONING" else "AUTO_FAST"
        )
        d = AutoModelDecision(
            chosen_model=selected,
            reason=reason,
            escalation=False,
            applied_constraints=constraints,
            tools_disabled=policy.tools_disabled,
        )
        out = _finalize_decision(inputs, d)
        _emit_audit(inputs, out)
        return out

    # Final fallback (escalation)
    fallback = _fallback_model(preferred_bucket, catalog, policy)
    if not fallback:
        raise ValueError("NO_ALLOWED_MODEL_AVAILABLE")

    d = AutoModelDecision(
        chosen_model=fallback,
        reason="NO_BUCKET_MODEL_FALLBACK",
        escalation=True,
        applied_constraints=constraints,
        tools_disabled=policy.tools_disabled,
    )
    out = _finalize_decision(inputs, d)
    _emit_audit(inputs, out)
    return out

