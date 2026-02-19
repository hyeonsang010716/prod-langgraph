"""Supervisor Multi-Agent 상담봇 POC (Dynamic Agent + Tool Registry)

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


# ===== CLI Helper Functions =====


def _print_agents(registry: AgentRegistry) -> None:
    agents = registry.list_all()
    if not agents:
        print("\n  (등록된 에이전트 없음)")
        return
    print("\n  등록된 에이전트:")
    for key, config in agents.items():
        tool_names = ", ".join(t.name for t in config.tools) or "없음"
        print(f"    [{key}] {config.description}")
        print(f"      도구: {tool_names}")


def _print_tools(registry: AgentRegistry, agent_key: str) -> None:
    config = registry.get(agent_key)
    if not config:
        print(f"  '{agent_key}' 에이전트를 찾을 수 없습니다.")
        return
    if not config.tools:
        print(f"  [{agent_key}] 등록된 도구 없음")
        return
    print(f"\n  [{agent_key}] 도구 목록:")
    for t in config.tools:
        print(f"    - {t.name}: {t.description}")
        print(f"      응답: {t.mock_response[:60]}...")


def _handle_add_agent(registry: AgentRegistry) -> None:
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
            f"고객의 {description} 관련 문의를 전문적으로 처리합니다.\n"
            "필요한 정보는 도구를 적극 활용하여 정확하게 안내하세요.\n"
            "항상 친절하고 전문적으로 한국어로 답변하세요."
        )
    registry.register(key, description, prompt)
    print(f"  '{key}' 에이전트가 추가되었습니다.")


def _handle_remove_agent(registry: AgentRegistry, arg: str) -> None:
    key = arg or input("  삭제할 에이전트 키: ").strip()
    if not key:
        print("  취소되었습니다.")
        return
    if registry.unregister(key):
        print(f"  '{key}' 에이전트가 삭제되었습니다.")
    else:
        print(f"  '{key}' 에이전트를 찾을 수 없습니다.")


def _handle_add_tool(registry: AgentRegistry) -> None:
    agent_key = input("  대상 에이전트 키: ").strip()
    if not agent_key or not registry.get(agent_key):
        print(f"  '{agent_key}' 에이전트를 찾을 수 없습니다.")
        return
    name = input("  도구 이름 (영문): ").strip()
    if not name:
        print("  취소되었습니다.")
        return
    description = input("  도구 설명 (LLM이 보는 설명): ").strip()
    if not description:
        print("  취소되었습니다.")
        return
    mock = input("  목업 응답 ({query}로 입력값 치환): ").strip()
    if not mock:
        mock = "'{query}'에 대한 처리가 완료되었습니다."
    registry.add_tool(agent_key, name, description, mock)
    print(f"  [{agent_key}] '{name}' 도구가 추가되었습니다.")


def _handle_remove_tool(registry: AgentRegistry) -> None:
    agent_key = input("  대상 에이전트 키: ").strip()
    tool_name = input("  삭제할 도구 이름: ").strip()
    if registry.remove_tool(agent_key, tool_name):
        print(f"  [{agent_key}] '{tool_name}' 도구가 삭제되었습니다.")
    else:
        print("  도구를 찾을 수 없습니다.")


def _print_help() -> None:
    print(
        "\n  === 상담 ===\n"
        "    메시지 입력        상담 대화\n"
        "\n"
        "  === 에이전트 관리 ===\n"
        "    /list              에이전트 + 도구 목록\n"
        "    /add               에이전트 추가\n"
        "    /remove <key>      에이전트 삭제\n"
        "\n"
        "  === 도구 관리 ===\n"
        "    /tools <key>       에이전트의 도구 상세\n"
        "    /add-tool          도구 추가\n"
        "    /remove-tool       도구 삭제\n"
        "\n"
        "  === 세션 ===\n"
        "    /new               새 세션 시작\n"
        "    /help              이 도움말"
    )


# ===== Main Loop =====


async def main() -> None:
    get_llm_manager().initialize()

    registry = AgentRegistry(persist_path=PERSIST_PATH)
    orchestrator = SupervisorGraphOrchestrator(registry)
    checkpointer = MemorySaver()
    await orchestrator.initialize(store=None, checkpointer=checkpointer)

    session_id = str(uuid.uuid4())
    is_first = True

    print("=" * 60)
    print("  Supervisor Multi-Agent 상담봇 (Dynamic Agent + Tool)")
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
                _handle_add_agent(registry)
            elif cmd == "/remove":
                _handle_remove_agent(registry, arg)
            elif cmd == "/tools":
                if arg:
                    _print_tools(registry, arg)
                else:
                    _print_agents(registry)
            elif cmd == "/add-tool":
                _handle_add_tool(registry)
            elif cmd == "/remove-tool":
                _handle_remove_tool(registry)
            elif cmd == "/new":
                session_id = str(uuid.uuid4())
                is_first = True
                print("  새 세션이 시작되었습니다.")
            elif cmd == "/help":
                _print_help()
            else:
                print(f"  알 수 없는 명령어: {cmd} (/help)")
            continue

        # ── 상담 채팅 (스트리밍) ──
        if is_first:
            stream = orchestrator.astream_invoke(user_input, session_id)
            is_first = False
        else:
            stream = orchestrator.astream_resume(session_id, user_input)

        agent_name = ""
        print()  # 응답 전 줄바꿈

        async for event in stream:
            event_type = event["type"]

            if event_type == "token":
                print(event["content"], end="", flush=True)

            elif event_type == "interrupt":
                agent_name = event.get("agent", "")

            elif event_type == "end":
                print()  # 토큰 출력 후 줄바꿈
                status = event["status"]
                exec_time = event["execution_time"]

                if status == "interrupted":
                    if agent_name:
                        print(f"  [{agent_name}] ({exec_time:.2f}s)")
                    else:
                        print(f"  ({exec_time:.2f}s)")

                elif status == "completed":
                    print(f"  ({exec_time:.2f}s)")
                    session_id = str(uuid.uuid4())
                    is_first = True
                    print("  (새 세션 준비 완료)")

            elif event_type == "error":
                print(f"\n[에러] {event['message']}")


if __name__ == "__main__":
    asyncio.run(main())
