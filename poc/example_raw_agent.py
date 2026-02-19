"""agents.json의 rental 설정을 일반 create_agent로 직접 작성한 경우.

동적 레지스트리 없이, 코드에 직접 정의하는 원래 형태입니다.

비교:
  - 동적(registry): JSON → ToolConfig → create_model() → StructuredTool → create_agent
  - 직접(이 파일):   Pydantic 클래스 → 함수 → StructuredTool → create_agent
"""

from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool
from app.core.llm_manager import ModelName, get_llm_manager


# ===== Tool Input Schemas (Pydantic) =====


class CheckRentalPriceInput(BaseModel):
    product_name: str = Field(description="조회할 렌탈 상품명 (예: 정수기, 공기청정기, 비데)")


class CheckRentalAvailabilityInput(BaseModel):
    product_name: str = Field(description="재고 확인할 렌탈 상품명")


# ===== Tool Functions =====


def check_rental_price(product_name: str) -> str:
    """렌탈 상품의 월 렌탈료를 조회합니다."""
    # 실제로는 DB 조회 또는 API 호출
    return f"'{product_name}' 상품의 월 렌탈료는 35,000원입니다. (약정기간: 36개월, 등록비: 무료)"


def check_rental_availability(product_name: str) -> str:
    """렌탈 상품의 재고 및 렌탈 가능 여부를 확인합니다."""
    # 실제로는 재고 시스템 조회
    return f"'{product_name}' 상품은 현재 렌탈 가능합니다. 설치 가능일: 3영업일 이내"


# ===== Tools =====

rental_tools = [
    StructuredTool.from_function(
        func=check_rental_price,
        name="check_rental_price",
        description="렌탈 상품의 월 렌탈료를 조회합니다.",
        args_schema=CheckRentalPriceInput,
    ),
    StructuredTool.from_function(
        func=check_rental_availability,
        name="check_rental_availability",
        description="렌탈 상품의 재고 및 렌탈 가능 여부를 확인합니다.",
        args_schema=CheckRentalAvailabilityInput,
    ),
]

# ===== Agent =====

model = get_llm_manager().get_model(ModelName.GPT_4O_MINI)

rental_agent = create_agent(
    model=model,
    tools=rental_tools,
    system_prompt=(
        "당신은 렌탈 상담 전문 에이전트입니다.\n"
        "고객의 렌탈 관련 문의를 전문적으로 처리합니다.\n"
        "필요한 정보는 도구를 적극 활용하여 정확하게 안내하세요.\n"
        "항상 친절하고 전문적으로 한국어로 답변하세요."
    ),
    name="rental",
)
