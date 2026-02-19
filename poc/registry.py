import json
import threading
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field as PydanticField


@dataclass
class AgentConfig:
    """에이전트 설정"""

    key: str
    description: str
    prompt: str


# ===== 기본 에이전트 (첫 실행 시 시드) =====

DEFAULT_AGENTS = [
    AgentConfig(
        key="rental",
        description="렌탈 계약, 렌탈 기간, 렌탈 비용, 렌탈 상품 관련 상담",
        prompt=(
            "당신은 렌탈 상담 전문 에이전트입니다.\n"
            "고객의 렌탈 관련 문의를 전문적으로 처리합니다.\n\n"
            "담당 업무:\n"
            "- 렌탈 상품 안내 및 추천\n"
            "- 렌탈 계약 조건 설명\n"
            "- 렌탈 기간 및 비용 안내\n"
            "- 렌탈 연장/해지 절차 안내\n\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
    ),
    AgentConfig(
        key="delivery",
        description="배송 조회, 배송 일정, 배송지 변경, 배송 관련 상담",
        prompt=(
            "당신은 배송 상담 전문 에이전트입니다.\n"
            "고객의 배송 관련 문의를 전문적으로 처리합니다.\n\n"
            "담당 업무:\n"
            "- 배송 상태 조회 및 안내\n"
            "- 배송 일정 확인\n"
            "- 배송지 변경 처리\n"
            "- 배송 지연/분실 대응\n\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
    ),
    AgentConfig(
        key="service",
        description="A/S 접수, 수리, 점검, 서비스 관련 상담",
        prompt=(
            "당신은 서비스(A/S) 상담 전문 에이전트입니다.\n"
            "고객의 서비스 관련 문의를 전문적으로 처리합니다.\n\n"
            "담당 업무:\n"
            "- A/S 접수 및 진행 상황 안내\n"
            "- 제품 수리/점검 상담\n"
            "- 서비스 이용 방법 안내\n"
            "- 보증 기간 및 무상/유상 서비스 안내\n\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
    ),
]


class AgentRegistry:
    """Thread-safe 에이전트 레지스트리

    에이전트 설정을 런타임에 동적으로 추가/삭제할 수 있습니다.
    변경 시 JSON 파일에 자동 저장되며, 시작 시 자동 로드됩니다.

    Usage:
        registry = AgentRegistry("poc/agents.json")
        registry.register("payment", "결제 관련 상담", "당신은 결제 전문 에이전트입니다...")
        registry.unregister("payment")
    """

    def __init__(self, persist_path: Optional[str] = None):
        self._agents: dict[str, AgentConfig] = {}
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        self._version = 0

        if self._persist_path and self._persist_path.exists():
            self._load()
        else:
            self._seed_defaults()

    def _seed_defaults(self) -> None:
        """기본 에이전트로 초기화합니다."""
        for config in DEFAULT_AGENTS:
            self._agents[config.key] = config
        self._save()

    def register(self, key: str, description: str, prompt: str) -> AgentConfig:
        """에이전트를 등록합니다. 이미 존재하면 덮어씁니다."""
        with self._lock:
            config = AgentConfig(key=key, description=description, prompt=prompt)
            self._agents[key] = config
            self._version += 1
            self._save()
            return config

    def unregister(self, key: str) -> bool:
        """에이전트를 삭제합니다. 성공 시 True."""
        with self._lock:
            if key in self._agents:
                del self._agents[key]
                self._version += 1
                self._save()
                return True
            return False

    def get(self, key: str) -> Optional[AgentConfig]:
        """에이전트 설정을 조회합니다."""
        return self._agents.get(key)

    def list_all(self) -> dict[str, AgentConfig]:
        """모든 에이전트를 조회합니다."""
        return dict(self._agents)

    @property
    def version(self) -> int:
        """레지스트리 버전 (변경 시마다 증가, 스키마 캐시 무효화에 사용)"""
        return self._version

    def get_descriptions(self) -> str:
        """Supervisor 프롬프트용 에이전트 설명을 생성합니다."""
        lines = []
        for key, config in self._agents.items():
            lines.append(f"- {key}: {config.description}")
        return "\n".join(lines)

    def build_route_schema(self) -> type[BaseModel]:
        """현재 등록된 에이전트 기반으로 라우팅 Pydantic 스키마를 동적 생성합니다.

        동적 Enum을 사용하여 with_structured_output에서 LLM이
        유효한 에이전트 키만 선택하도록 강제합니다.
        """
        choices = {k: k for k in self._agents}
        choices["end"] = "end"

        RouteEnum = Enum("RouteEnum", choices)

        class RouteDecision(BaseModel):
            next: RouteEnum = PydanticField(
                description=(
                    "라우팅할 에이전트 키 또는 종료(end). "
                    f"선택지: {', '.join(choices)}"
                )
            )

        return RouteDecision

    def _save(self) -> None:
        if not self._persist_path:
            return
        data = {k: asdict(v) for k, v in self._agents.items()}
        self._persist_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2)
        )

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        data = json.loads(self._persist_path.read_text())
        for key, config_dict in data.items():
            self._agents[key] = AgentConfig(**config_dict)
