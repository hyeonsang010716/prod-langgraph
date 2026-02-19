"""Supervisor Multi-Agent 상담봇 POC (Dynamic Agent Registry)

실행:
    uv run python -m poc.main
"""

import asyncio
import uuid
from pathlib import Path

from langgraph.checkpoint.memory import MemorySaver

from app.core.llm_manager import get_llm_manager
from poc.graph import SupervisorGraphOrchestrator
from poc.registry import AgentRegistry

PERSIST_PATH = str(Path(__file__).parent / "agents.json")


def _print_agents(registry: AgentRegistry) -> None:
    agents = registry.list_all()
    if not agents:
        print("\n  (등록된 에이전트 없음)")
        return
    print("\n  등록된 에이전트:")
    for key, config in agents.items():
        print(f"    - {key}: {config.description}")


def _handle_add(registry: AgentRegistry) -> None:
    key = input("  에이전트 키 (영문): ").strip()
    if not key:
        print("  취소되었습니다.")
        return
    description = input("  설명: ").strip()
    if not description:
        print("  취소되었습니다.")
        return
    custom = input("  커스텀 프롬프트 (빈칸=자동생성): ").strip()
    if custom:
        prompt = custom
    else:
        prompt = (
            f"당신은 {description} 전문 에이전트입니다.\n"
            f"고객의 {description} 관련 문의를 전문적으로 처리합니다.\n\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        )
    registry.register(key, description, prompt)
    print(f"  '{key}' 에이전트가 추가되었습니다.")


def _handle_remove(registry: AgentRegistry, arg: str) -> None:
    key = arg or input("  삭제할 에이전트 키: ").strip()
    if not key:
        print("  취소되었습니다.")
        return
    if registry.unregister(key):
        print(f"  '{key}' 에이전트가 삭제되었습니다.")
    else:
        print(f"  '{key}' 에이전트를 찾을 수 없습니다.")


def _print_help() -> None:
    print(
        "\n  명령어:\n"
        "    /list          등록된 에이전트 목록\n"
        "    /add           새 에이전트 추가\n"
        "    /remove <key>  에이전트 삭제\n"
        "    /new           새 세션 시작\n"
        "    /help          이 도움말"
    )


async def main() -> None:
    # 초기화
    get_llm_manager().initialize()

    registry = AgentRegistry(persist_path=PERSIST_PATH)
    orchestrator = SupervisorGraphOrchestrator(registry)
    checkpointer = MemorySaver()
    await orchestrator.initialize(store=None, checkpointer=checkpointer)

    session_id = str(uuid.uuid4())
    is_first = True

    print("=" * 60)
    print("  Supervisor Multi-Agent 상담봇 (Dynamic)")
    print("  /help 로 명령어 확인  |  Ctrl+C 종료")
    print("=" * 60)
    _print_agents(registry)

    while True:
        try:
            user_input = input("\n[사용자] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n프로그램을 종료합니다.")
            break

        if not user_input:
            continue

        # ── 슬래시 명령어 ──
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/list":
                _print_agents(registry)
            elif cmd == "/add":
                _handle_add(registry)
            elif cmd == "/remove":
                _handle_remove(registry, arg)
            elif cmd == "/new":
                session_id = str(uuid.uuid4())
                is_first = True
                print("  새 세션이 시작되었습니다.")
            elif cmd == "/help":
                _print_help()
            else:
                print(f"  알 수 없는 명령어: {cmd} (/help)")
            continue

        # ── 상담 채팅 ──
        if is_first:
            result = await orchestrator.ainvoke(user_input, session_id)
            is_first = False
        else:
            result = await orchestrator.aresume(session_id, user_input)

        status = result["status"]

        if status == "interrupted":
            for intr in result["interrupt_info"].get("interrupts", []):
                value = intr["value"]
                agent = value.get("agent", "unknown")
                message = value.get("message", "")
                print(f"\n[{agent}] {message}")
            print(f"  ({result['execution_time']:.2f}s)")

        elif status == "completed":
            print(f"\n[시스템] {result['answer']}")
            print(f"  ({result['execution_time']:.2f}s)")
            # 종료 후 자동으로 새 세션 준비
            session_id = str(uuid.uuid4())
            is_first = True
            print("  (새 세션이 자동 시작되었습니다. 바로 다음 문의를 입력하세요.)")

        elif status == "error":
            print(f"\n[에러] {result['message']}")


if __name__ == "__main__":
    asyncio.run(main())
