import json
import threading
from dataclasses import dataclass, asdict, field
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field as PydanticField


# ===== Config Dataclasses =====


@dataclass
class ToolParam:
    """도구 파라미터 정의

    name: 파라미터 이름 (영문, LLM이 JSON key로 사용)
    description: LLM이 파라미터를 채울 때 참고하는 설명
    """

    name: str
    description: str


@dataclass
class ToolConfig:
    """에이전트 도구 설정

    name: 도구 이름 (영문, 에이전트 내 고유)
    description: LLM이 도구를 선택할 때 참고하는 설명
    mock_response: POC용 목업 응답 (파라미터 이름으로 치환 가능, 예: {phone_number})
    parameters: 도구 입력 파라미터 목록 (빈 리스트면 query: str 단일 파라미터)
    """

    name: str
    description: str
    mock_response: str
    parameters: list[ToolParam] = field(default_factory=list)


@dataclass
class AgentConfig:
    """에이전트 설정

    key: 에이전트 고유 키 (영문)
    description: Supervisor가 라우팅 시 참고하는 설명
    prompt: 에이전트 시스템 프롬프트
    tools: 에이전트가 사용할 도구 목록
    """

    key: str
    description: str
    prompt: str
    tools: list[ToolConfig] = field(default_factory=list)


# ===== Default Agents =====

DEFAULT_AGENTS = [
    AgentConfig(
        key="rental",
        description="렌탈 계약, 렌탈 기간, 렌탈 비용, 렌탈 상품 관련 상담",
        prompt=(
            "당신은 렌탈 상담 전문 에이전트입니다.\n"
            "고객의 렌탈 관련 문의를 전문적으로 처리합니다.\n"
            "필요한 정보는 도구를 적극 활용하여 정확하게 안내하세요.\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
        tools=[
            ToolConfig(
                name="check_rental_price",
                description="렌탈 상품의 월 렌탈료를 조회합니다.",
                mock_response="'{product_name}' 상품의 월 렌탈료는 35,000원입니다. (약정기간: 36개월, 등록비: 무료)",
                parameters=[
                    ToolParam(name="product_name", description="조회할 렌탈 상품명 (예: 정수기, 공기청정기, 비데)"),
                ],
            ),
            ToolConfig(
                name="check_rental_availability",
                description="렌탈 상품의 재고 및 렌탈 가능 여부를 확인합니다.",
                mock_response="'{product_name}' 상품은 현재 렌탈 가능합니다. 설치 가능일: 3영업일 이내",
                parameters=[
                    ToolParam(name="product_name", description="재고 확인할 렌탈 상품명"),
                ],
            ),
        ],
    ),
    AgentConfig(
        key="delivery",
        description="배송 조회, 배송 일정, 배송지 변경, 배송 관련 상담",
        prompt=(
            "당신은 배송 상담 전문 에이전트입니다.\n"
            "고객의 배송 관련 문의를 전문적으로 처리합니다.\n"
            "필요한 정보는 도구를 적극 활용하여 정확하게 안내하세요.\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
        tools=[
            ToolConfig(
                name="track_delivery",
                description="주문번호로 배송 상태를 조회합니다.",
                mock_response="주문 '{order_id}'의 배송 상태: 배송중 (현재 OO물류센터 → 고객 지역 배송 예정)",
                parameters=[
                    ToolParam(name="order_id", description="조회할 주문번호 (예: ORD-2024-00123)"),
                ],
            ),
            ToolConfig(
                name="change_delivery_address",
                description="배송지 주소를 변경합니다.",
                mock_response="주문 '{order_id}'의 배송지가 '{new_address}'(으)로 변경 요청되었습니다. 확인 후 반영됩니다.",
                parameters=[
                    ToolParam(name="order_id", description="주문번호"),
                    ToolParam(name="new_address", description="변경할 새 배송지 주소"),
                ],
            ),
        ],
    ),
    AgentConfig(
        key="service",
        description="A/S 접수, 수리, 점검, 서비스 관련 상담",
        prompt=(
            "당신은 서비스(A/S) 상담 전문 에이전트입니다.\n"
            "고객의 서비스 관련 문의를 전문적으로 처리합니다.\n"
            "필요한 정보는 도구를 적극 활용하여 정확하게 안내하세요.\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
        tools=[
            ToolConfig(
                name="submit_service_request",
                description="A/S 요청을 접수합니다.",
                mock_response="A/S 접수 완료 — 접수번호: AS-2024-00456. 제품: '{product_name}', 증상: '{symptom}'. 엔지니어 방문 예정일: 2~3영업일 이내",
                parameters=[
                    ToolParam(name="product_name", description="A/S 대상 제품명 (예: 정수기, 공기청정기)"),
                    ToolParam(name="symptom", description="고장 증상 상세 설명"),
                ],
            ),
            ToolConfig(
                name="check_warranty",
                description="제품의 보증기간을 확인합니다.",
                mock_response="'{serial_number}' 제품 보증 상태: 무상 보증기간 내 (만료일: 2025-12-31)",
                parameters=[
                    ToolParam(name="serial_number", description="제품 시리얼번호 또는 제품명"),
                ],
            ),
        ],
    ),
    AgentConfig(
        key="customer",
        description="고객정보 조회, 본인확인, 계약자 확인, 연락처 입력, 생년월일 인증 관련 상담",
        prompt=(
            "당신은 고객정보 조회 및 본인확인 전문 에이전트입니다.\n"
            "고객의 계약자 정보를 확인하고 본인 인증을 처리합니다.\n"
            "연락처가 미입력된 경우 전화번호 입력을 안내하고, "
            "생년월일 불일치 시 재입력을 요청합니다.\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        ),
        tools=[
            ToolConfig(
                name="lookup_customer_by_phone",
                description="계약자의 전화번호로 고객정보를 조회합니다. 연락처가 미입력된 경우 전화번호를 입력받아 조회합니다.",
                mock_response=(
                    "전화번호 '{phone_number}'(으)로 조회한 결과, "
                    "계약자명: 홍길동, 계약번호: CT-2024-00123, "
                    "계약상품: 정수기 렌탈 (36개월). 고객정보 확인이 완료되었습니다."
                ),
                parameters=[
                    ToolParam(name="phone_number", description="계약자 전화번호 (예: 01012345678)"),
                ],
            ),
            ToolConfig(
                name="verify_customer_birthday",
                description="계약자의 생년월일로 본인확인을 진행합니다. 생년월일이 불일치하면 재입력을 요청합니다.",
                mock_response=(
                    "생년월일 '{birthday}'(으)로 본인확인을 진행합니다. "
                    "본인확인이 완료되었습니다. 계약자: 홍길동님, 인증 성공."
                ),
                parameters=[
                    ToolParam(name="birthday", description="계약자 생년월일 6자리 (예: 810902)"),
                ],
            ),
        ],
    ),
]


class AgentRegistry:
    """Thread-safe 에이전트+도구 레지스트리

    에이전트와 도구를 런타임에 동적으로 추가/삭제할 수 있습니다.
    변경 시 JSON 파일에 자동 저장되며, 시작 시 자동 로드됩니다.
    version 속성으로 변경 감지 → 서브그래프 캐시 무효화에 사용합니다.
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

    # ── Agent CRUD ──

    def register(self, key: str, description: str, prompt: str) -> AgentConfig:
        """에이전트를 등록합니다. 이미 존재하면 덮어씁니다 (도구는 유지)."""
        with self._lock:
            existing_tools = []
            if key in self._agents:
                existing_tools = self._agents[key].tools
            config = AgentConfig(
                key=key, description=description, prompt=prompt, tools=existing_tools
            )
            self._agents[key] = config
            self._version += 1
            self._save()
            return config

    def unregister(self, key: str) -> bool:
        """에이전트를 삭제합니다 (도구 포함)."""
        with self._lock:
            if key in self._agents:
                del self._agents[key]
                self._version += 1
                self._save()
                return True
            return False

    def get(self, key: str) -> Optional[AgentConfig]:
        return self._agents.get(key)

    def list_all(self) -> dict[str, AgentConfig]:
        return dict(self._agents)

    # ── Tool CRUD ──

    def add_tool(
        self,
        agent_key: str,
        name: str,
        description: str,
        mock_response: str,
        parameters: list[ToolParam] | None = None,
    ) -> ToolConfig:
        """에이전트에 도구를 추가합니다."""
        with self._lock:
            config = self._agents.get(agent_key)
            if not config:
                raise KeyError(f"'{agent_key}' 에이전트가 존재하지 않습니다.")
            # 같은 이름 도구가 있으면 교체
            config.tools = [t for t in config.tools if t.name != name]
            tool = ToolConfig(
                name=name,
                description=description,
                mock_response=mock_response,
                parameters=parameters or [],
            )
            config.tools.append(tool)
            self._version += 1
            self._save()
            return tool

    def remove_tool(self, agent_key: str, tool_name: str) -> bool:
        """에이전트에서 도구를 삭제합니다."""
        with self._lock:
            config = self._agents.get(agent_key)
            if not config:
                return False
            original_len = len(config.tools)
            config.tools = [t for t in config.tools if t.name != tool_name]
            if len(config.tools) < original_len:
                self._version += 1
                self._save()
                return True
            return False

    # ── Query Helpers ──

    @property
    def version(self) -> int:
        """레지스트리 버전 (변경 시마다 증가, 캐시 무효화에 사용)"""
        return self._version

    def get_descriptions(self) -> str:
        """Supervisor 프롬프트용 에이전트 설명을 생성합니다."""
        lines = []
        for key, config in self._agents.items():
            tool_names = ", ".join(t.name for t in config.tools) or "없음"
            lines.append(f"- {key}: {config.description} (도구: {tool_names})")
        return "\n".join(lines)

    def build_route_schema(self) -> type[BaseModel]:
        """현재 등록된 에이전트 기반으로 라우팅 스키마를 동적 생성합니다."""
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

    # ── Persistence ──

    def _seed_defaults(self) -> None:
        for config in DEFAULT_AGENTS:
            self._agents[config.key] = config
        self._save()

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
            tools_raw = config_dict.pop("tools", [])
            tools = []
            for t in tools_raw:
                params_raw = t.pop("parameters", [])
                params = [ToolParam(**p) for p in params_raw]
                tools.append(ToolConfig(**t, parameters=params))
            self._agents[key] = AgentConfig(**config_dict, tools=tools)
