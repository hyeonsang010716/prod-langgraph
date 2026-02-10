import json
from datetime import datetime
from typing import Any, Optional


# ============================================================
# 유틸리티 함수
# ============================================================

def _safe_str(value: Any, max_len: int = 200) -> str:
    """값을 안전하게 문자열로 변환, 길이 제한"""
    try:
        s = str(value)
        if len(s) > max_len:
            return s[:max_len] + f"... (총 {len(s)}자)"
        return s
    except Exception as e:
        return f"<표시 불가: {e}>"


def _safe_json(value: Any, indent: int = 2) -> str:
    """JSON으로 예쁘게 출력 시도, 실패 시 str"""
    try:
        return json.dumps(value, indent=indent, default=str, ensure_ascii=False)
    except Exception:
        return _safe_str(value, max_len=500)


def _format_separator(title: str, char: str = "=", width: int = 80) -> str:
    return f"\n{char * width}\n  {title}\n{char * width}"


def _format_sub_separator(title: str, char: str = "-", width: int = 60) -> str:
    return f"\n  {char * width}\n  {title}\n  {char * width}"


# ============================================================
# 1. Config 정보 추출
# ============================================================

def print_config_info(checkpoint_tuple) -> None:
    """CheckpointTuple.config에서 추출 가능한 모든 정보 출력"""
    print(_format_sub_separator("📋 CONFIG 정보"))
    
    config = checkpoint_tuple.config
    if not config:
        print("    (config 없음)")
        return
    
    configurable = config.get("configurable", {})
    
    # 핵심 식별자
    print(f"    thread_id       : {configurable.get('thread_id', 'N/A')}")
    print(f"    checkpoint_id   : {configurable.get('checkpoint_id', 'N/A')}")
    print(f"    checkpoint_ns   : {configurable.get('checkpoint_ns', '(root)')}")
    
    # configurable 내 기타 키 (커스텀 설정 등)
    known_keys = {'thread_id', 'checkpoint_id', 'checkpoint_ns'}
    extra_keys = set(configurable.keys()) - known_keys
    if extra_keys:
        print(f"    기타 configurable 키:")
        for k in sorted(extra_keys):
            print(f"      {k}: {_safe_str(configurable[k])}")
    
    # config 최상위 레벨 기타 키
    config_extra = set(config.keys()) - {'configurable'}
    if config_extra:
        print(f"    config 기타 키:")
        for k in sorted(config_extra):
            print(f"      {k}: {_safe_str(config[k])}")


# ============================================================
# 2. Metadata 정보 추출
# ============================================================

def print_metadata_info(checkpoint_tuple) -> None:
    """CheckpointTuple.metadata에서 추출 가능한 모든 정보 출력"""
    print(_format_sub_separator("📝 METADATA 정보"))
    
    metadata = checkpoint_tuple.metadata
    if not metadata:
        print("    (metadata 없음)")
        return
    
    # source: 체크포인트 생성 원인
    source = metadata.get('source', 'N/A')
    source_desc = {
        'input': '사용자 입력 (invoke/stream 호출)',
        'loop': 'Pregel 루프 내부 실행',
        'update': '수동 상태 업데이트 (update_state)',
        'fork': '다른 체크포인트에서 분기(fork)',
    }
    print(f"    source          : {source} → {source_desc.get(source, '알 수 없음')}")
    
    # step: 실행 단계
    step = metadata.get('step', 'N/A')
    step_desc = ""
    if step == -1:
        step_desc = " (초기 input 체크포인트)"
    elif step == 0:
        step_desc = " (첫 번째 loop 체크포인트)"
    print(f"    step            : {step}{step_desc}")
    
    # writes: 이 체크포인트에서 기록된 데이터
    writes = metadata.get('writes', None)
    if writes is not None:
        print(f"    writes          :")
        if isinstance(writes, dict):
            for node_name, write_data in writes.items():
                print(f"      노드 '{node_name}':")
                if isinstance(write_data, dict):
                    for k, v in write_data.items():
                        print(f"        {k}: {_safe_str(v, max_len=150)}")
                elif isinstance(write_data, list):
                    for i, item in enumerate(write_data):
                        print(f"        [{i}]: {_safe_str(item, max_len=150)}")
                else:
                    print(f"        {_safe_str(write_data, max_len=150)}")
        else:
            print(f"      {_safe_str(writes, max_len=300)}")
    else:
        print(f"    writes          : None")
    
    # parents: 부모 체크포인트 ID 매핑
    parents = metadata.get('parents', {})
    if parents:
        print(f"    parents         :")
        for ns, pid in parents.items():
            ns_label = ns if ns else "(root)"
            print(f"      namespace '{ns_label}' → checkpoint_id: {pid}")
    else:
        print(f"    parents         : (없음 - 루트 체크포인트)")
    
    # metadata 내 기타 키 (사용자 커스텀 메타데이터 등)
    known_meta_keys = {'source', 'step', 'writes', 'parents'}
    extra_meta = set(metadata.keys()) - known_meta_keys
    if extra_meta:
        print(f"    기타 metadata 키:")
        for k in sorted(extra_meta):
            print(f"      {k}: {_safe_str(metadata[k], max_len=200)}")


# ============================================================
# 3. Checkpoint (상태 스냅샷) 정보 추출
# ============================================================

def print_checkpoint_info(checkpoint_tuple) -> None:
    """CheckpointTuple.checkpoint에서 추출 가능한 모든 정보 출력"""
    print(_format_sub_separator("💾 CHECKPOINT (상태 스냅샷) 정보"))
    
    checkpoint = checkpoint_tuple.checkpoint
    if not checkpoint:
        print("    (checkpoint 없음)")
        return
    
    # 기본 정보
    print(f"    v (버전)        : {checkpoint.get('v', 'N/A')}")
    print(f"    id              : {checkpoint.get('id', 'N/A')}")
    ts = checkpoint.get('ts', 'N/A')
    print(f"    ts (타임스탬프)  : {ts}")
    if ts and ts != 'N/A':
        try:
            dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
            print(f"    ts (로컬 변환)  : {dt.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        except Exception:
            pass
    
    # ── channel_values: 가장 중요한 상태 데이터 ──
    channel_values = checkpoint.get('channel_values', {})
    print(f"\n    📊 channel_values ({len(channel_values)}개 채널):")
    
    for ch_name, ch_value in channel_values.items():
        print(f"\n      채널 '{ch_name}':")
        
        # messages 채널 (가장 흔한 케이스) 특별 처리
        if ch_name == 'messages' and isinstance(ch_value, list):
            print(f"        메시지 수: {len(ch_value)}")
            for i, msg in enumerate(ch_value):
                _print_message_detail(msg, indent=8, index=i)
        
        # 딕셔너리 채널
        elif isinstance(ch_value, dict):
            for k, v in ch_value.items():
                print(f"        {k}: {_safe_str(v, max_len=150)}")
        
        # 리스트 채널 (messages가 아닌)
        elif isinstance(ch_value, list):
            print(f"        항목 수: {len(ch_value)}")
            for i, item in enumerate(ch_value[:10]):  # 최대 10개
                print(f"        [{i}]: {_safe_str(item, max_len=150)}")
            if len(ch_value) > 10:
                print(f"        ... 외 {len(ch_value) - 10}개 더")
        
        # 기타
        else:
            print(f"        값: {_safe_str(ch_value, max_len=200)}")
            print(f"        타입: {type(ch_value).__name__}")
    
    # ── channel_versions ──
    channel_versions = checkpoint.get('channel_versions', {})
    if channel_versions:
        print(f"\n    📌 channel_versions ({len(channel_versions)}개):")
        for ch_name, version in sorted(channel_versions.items()):
            print(f"      {ch_name}: {version}")
    
    # ── versions_seen: 각 노드가 본 채널 버전 ──
    versions_seen = checkpoint.get('versions_seen', {})
    if versions_seen:
        print(f"\n    👁️  versions_seen ({len(versions_seen)}개 노드):")
        for node_id, ch_versions in versions_seen.items():
            node_label = node_id if node_id else "__input__"
            print(f"      노드 '{node_label}':")
            if isinstance(ch_versions, dict):
                for ch_name, ver in sorted(ch_versions.items()):
                    print(f"        {ch_name}: {ver}")
            else:
                print(f"        {_safe_str(ch_versions)}")
    
    # ── updated_channels ──
    updated_channels = checkpoint.get('updated_channels')
    if updated_channels is not None:
        print(f"\n    🔄 updated_channels: {updated_channels}")
    
    # ── checkpoint 내 기타 키 ──
    known_cp_keys = {'v', 'id', 'ts', 'channel_values', 'channel_versions', 
                     'versions_seen', 'updated_channels'}
    extra_cp = set(checkpoint.keys()) - known_cp_keys
    if extra_cp:
        print(f"\n    기타 checkpoint 키:")
        for k in sorted(extra_cp):
            print(f"      {k}: {_safe_str(checkpoint[k], max_len=200)}")


def _print_message_detail(msg: Any, indent: int = 8, index: int = 0) -> None:
    """메시지 객체의 상세 정보 출력 (HumanMessage, AIMessage, ToolMessage 등)"""
    pad = " " * indent
    
    # LangChain BaseMessage 객체인 경우
    if hasattr(msg, 'type') and hasattr(msg, 'content'):
        msg_type = getattr(msg, 'type', '?')
        content = getattr(msg, 'content', '')
        msg_id = getattr(msg, 'id', None)
        name = getattr(msg, 'name', None)
        
        type_emoji = {
            'human': '👤', 'ai': '🤖', 'system': '⚙️', 
            'tool': '🔧', 'function': '📦'
        }.get(msg_type, '📨')
        
        print(f"{pad}[{index}] {type_emoji} {msg_type}")
        
        # content 처리 (문자열 또는 멀티모달 리스트)
        if isinstance(content, str):
            content_preview = content[:200] + ('...' if len(content) > 200 else '')
            print(f"{pad}    content: {content_preview}")
            if len(content) > 200:
                print(f"{pad}    content 전체 길이: {len(content)}자")
        elif isinstance(content, list):
            # 멀티모달 콘텐츠 (텍스트 + 이미지 등)
            print(f"{pad}    content (멀티모달, {len(content)}개 블록):")
            for j, block in enumerate(content):
                if isinstance(block, dict):
                    block_type = block.get('type', '?')
                    if block_type == 'text':
                        print(f"{pad}      [{j}] text: {_safe_str(block.get('text', ''), 100)}")
                    elif block_type == 'image_url':
                        print(f"{pad}      [{j}] image_url: {_safe_str(block.get('image_url', {}).get('url', ''), 80)}")
                    elif block_type == 'tool_use':
                        print(f"{pad}      [{j}] tool_use: name={block.get('name')}, id={block.get('id')}")
                        print(f"{pad}          input: {_safe_str(block.get('input', {}), 150)}")
                    else:
                        print(f"{pad}      [{j}] {block_type}: {_safe_str(block, 100)}")
                else:
                    print(f"{pad}      [{j}]: {_safe_str(block, 100)}")
        
        if msg_id:
            print(f"{pad}    id: {msg_id}")
        if name:
            print(f"{pad}    name: {name}")
        
        # AI 메시지 특수 필드
        if msg_type == 'ai':
            # tool_calls
            tool_calls = getattr(msg, 'tool_calls', None)
            if tool_calls:
                print(f"{pad}    tool_calls ({len(tool_calls)}개):")
                for tc in tool_calls:
                    if isinstance(tc, dict):
                        print(f"{pad}      - name: {tc.get('name')}, id: {tc.get('id')}")
                        print(f"{pad}        args: {_safe_str(tc.get('args', {}), 150)}")
                    else:
                        print(f"{pad}      - {_safe_str(tc, 150)}")
            
            # invalid_tool_calls
            invalid_tc = getattr(msg, 'invalid_tool_calls', None)
            if invalid_tc:
                print(f"{pad}    invalid_tool_calls: {_safe_str(invalid_tc, 150)}")
            
            # usage_metadata (토큰 사용량)
            usage = getattr(msg, 'usage_metadata', None)
            if usage:
                print(f"{pad}    usage_metadata:")
                if isinstance(usage, dict):
                    for k, v in usage.items():
                        print(f"{pad}      {k}: {v}")
                else:
                    print(f"{pad}      {_safe_str(usage, 150)}")
            
            # response_metadata
            resp_meta = getattr(msg, 'response_metadata', None)
            if resp_meta:
                print(f"{pad}    response_metadata:")
                if isinstance(resp_meta, dict):
                    for k, v in resp_meta.items():
                        print(f"{pad}      {k}: {_safe_str(v, 100)}")
                else:
                    print(f"{pad}      {_safe_str(resp_meta, 150)}")
        
        # Tool 메시지 특수 필드
        if msg_type == 'tool':
            tool_call_id = getattr(msg, 'tool_call_id', None)
            if tool_call_id:
                print(f"{pad}    tool_call_id: {tool_call_id}")
            status = getattr(msg, 'status', None)
            if status:
                print(f"{pad}    status: {status}")
            artifact = getattr(msg, 'artifact', None)
            if artifact:
                print(f"{pad}    artifact: {_safe_str(artifact, 100)}")
        
        # additional_kwargs
        additional = getattr(msg, 'additional_kwargs', None)
        if additional:
            print(f"{pad}    additional_kwargs: {_safe_str(additional, 150)}")
    
    # 딕셔너리 형태의 메시지
    elif isinstance(msg, dict):
        print(f"{pad}[{index}] (dict) type={msg.get('type', '?')}")
        print(f"{pad}    content: {_safe_str(msg.get('content', ''), 200)}")
        for k, v in msg.items():
            if k not in ('type', 'content'):
                print(f"{pad}    {k}: {_safe_str(v, 100)}")
    
    # 기타
    else:
        print(f"{pad}[{index}] ({type(msg).__name__}): {_safe_str(msg, 200)}")


# ============================================================
# 4. Pending Writes 정보 추출
# ============================================================

def print_pending_writes_info(checkpoint_tuple) -> None:
    """CheckpointTuple.pending_writes에서 추출 가능한 모든 정보 출력"""
    print(_format_sub_separator("✏️  PENDING WRITES (대기 중인 쓰기)"))
    
    pending_writes = getattr(checkpoint_tuple, 'pending_writes', None)
    if not pending_writes:
        print("    (pending_writes 없음)")
        return
    
    print(f"    총 {len(pending_writes)}개 pending write:")
    for i, write in enumerate(pending_writes):
        print(f"\n    [{i}]")
        if isinstance(write, tuple) and len(write) >= 3:
            task_id, channel, value = write[0], write[1], write[2]
            print(f"      task_id : {task_id}")
            print(f"      channel : {channel}")
            print(f"      value   : {_safe_str(value, max_len=200)}")
            if len(write) > 3:
                print(f"      extra   : {_safe_str(write[3:], max_len=100)}")
        else:
            print(f"      raw: {_safe_str(write, max_len=300)}")


# ============================================================
# 5. Parent Config 정보 추출
# ============================================================

def print_parent_config_info(checkpoint_tuple) -> None:
    """CheckpointTuple.parent_config 정보 출력"""
    print(_format_sub_separator("🔗 PARENT CONFIG (부모 체크포인트)"))
    
    parent_config = getattr(checkpoint_tuple, 'parent_config', None)
    if not parent_config:
        print("    (parent_config 없음 - 이것이 첫 번째 체크포인트)")
        return
    
    parent_configurable = parent_config.get('configurable', {})
    print(f"    parent thread_id      : {parent_configurable.get('thread_id', 'N/A')}")
    print(f"    parent checkpoint_id  : {parent_configurable.get('checkpoint_id', 'N/A')}")
    print(f"    parent checkpoint_ns  : {parent_configurable.get('checkpoint_ns', '(root)')}")


# ============================================================
# 6. CheckpointTuple 객체 자체의 추가 속성 탐색
# ============================================================

def print_extra_attributes(checkpoint_tuple) -> None:
    """CheckpointTuple 객체에 알려지지 않은 추가 속성이 있는지 탐색"""
    print(_format_sub_separator("🔍 추가 속성 탐색"))
    
    known_attrs = {'config', 'checkpoint', 'metadata', 'parent_config', 'pending_writes'}
    
    all_attrs = set()
    # NamedTuple의 필드
    if hasattr(checkpoint_tuple, '_fields'):
        all_attrs.update(checkpoint_tuple._fields)
    # 일반 속성
    for attr in dir(checkpoint_tuple):
        if not attr.startswith('_'):
            all_attrs.add(attr)
    
    extra_attrs = all_attrs - known_attrs - {'count', 'index'}  # 기본 tuple 메서드 제외
    
    if extra_attrs:
        print(f"    발견된 추가 속성:")
        for attr in sorted(extra_attrs):
            try:
                value = getattr(checkpoint_tuple, attr)
                if not callable(value):
                    print(f"      {attr}: {_safe_str(value, max_len=200)}")
            except Exception as e:
                print(f"      {attr}: <접근 불가: {e}>")
    else:
        print("    (추가 속성 없음)")


# ============================================================
# 7. 요약 통계
# ============================================================

def print_summary_stats(checkpoints: list) -> None:
    """전체 체크포인트 목록에 대한 요약 통계"""
    print(_format_separator("📊 전체 요약 통계", "█"))
    
    if not checkpoints:
        print("  (체크포인트 없음)")
        return
    
    print(f"  총 체크포인트 수: {len(checkpoints)}개")
    
    # thread_id 별 그룹화
    threads = {}
    for ct in checkpoints:
        tid = ct.config.get('configurable', {}).get('thread_id', 'unknown')
        threads.setdefault(tid, []).append(ct)
    
    print(f"  고유 thread 수  : {len(threads)}개")
    for tid, cts in threads.items():
        print(f"    thread '{tid}': {len(cts)}개 체크포인트")
    
    # source 분포
    sources = {}
    for ct in checkpoints:
        src = (ct.metadata or {}).get('source', 'unknown')
        sources[src] = sources.get(src, 0) + 1
    print(f"\n  source 분포:")
    for src, count in sorted(sources.items()):
        print(f"    {src}: {count}개")
    
    # step 범위
    steps = [
        (ct.metadata or {}).get('step', None) 
        for ct in checkpoints 
        if (ct.metadata or {}).get('step') is not None
    ]
    if steps:
        print(f"\n  step 범위: {min(steps)} ~ {max(steps)}")
    
    # 메시지 수 추이
    print(f"\n  메시지 수 추이 (최신→과거):")
    for ct in checkpoints:
        cp = ct.checkpoint or {}
        cv = cp.get('channel_values', {})
        msgs = cv.get('messages', [])
        step = (ct.metadata or {}).get('step', '?')
        source = (ct.metadata or {}).get('source', '?')
        cpid = ct.config.get('configurable', {}).get('checkpoint_id', '?')[:12]
        print(f"    step={step:>3}, source={source:<6}, msgs={len(msgs):>3}, id={cpid}...")
    
    # 타임스탬프 범위
    timestamps = []
    for ct in checkpoints:
        ts = (ct.checkpoint or {}).get('ts')
        if ts:
            timestamps.append(ts)
    if timestamps:
        print(f"\n  시간 범위:")
        print(f"    최초: {min(timestamps)}")
        print(f"    최근: {max(timestamps)}")


# ============================================================
# 메인 함수 (비동기)
# ============================================================

async def inspect_all_checkpoints(
    checkpointer,
    config: Optional[dict] = None,
    limit: int = 10,
    show_messages: bool = True,
    show_channel_versions: bool = True,
    show_versions_seen: bool = True,
) -> list:
    """
    체크포인터에서 추출 가능한 모든 정보를 출력합니다.
    
    Args:
        checkpointer: BaseCheckpointSaver 인스턴스 (PostgresSaver, InMemorySaver 등)
        config: 필터링 설정. None이면 모든 thread 조회.
                예: {"configurable": {"thread_id": "my-thread"}}
        limit: 최대 조회 수
        show_messages: 메시지 상세 출력 여부
        show_channel_versions: channel_versions 출력 여부
        show_versions_seen: versions_seen 출력 여부
    
    Returns:
        조회된 CheckpointTuple 리스트
    """
    print(_format_separator(
        f"🔎 LangGraph Checkpoint Inspector (limit={limit})", "█"
    ))
    
    if config:
        tid = config.get('configurable', {}).get('thread_id', '(미지정)')
        print(f"  조회 대상 thread_id: {tid}")
    else:
        print(f"  조회 대상: 모든 thread (config=None)")
    
    checkpoints = []
    idx = 0
    
    async for checkpoint_tuple in checkpointer.alist(config, limit=limit):
        checkpoints.append(checkpoint_tuple)
        
        print(_format_separator(
            f"체크포인트 #{idx + 1}", "━"
        ))
        
        # 1. Config 정보
        print_config_info(checkpoint_tuple)
        
        # 2. Metadata 정보
        print_metadata_info(checkpoint_tuple)
        
        # 3. Checkpoint (상태 스냅샷) 정보
        print_checkpoint_info(checkpoint_tuple)
        
        # 4. Pending Writes
        print_pending_writes_info(checkpoint_tuple)
        
        # 5. Parent Config
        print_parent_config_info(checkpoint_tuple)
        
        # 6. 추가 속성 탐색
        print_extra_attributes(checkpoint_tuple)
        
        idx += 1
    
    # 7. 전체 요약 통계
    print_summary_stats(checkpoints)
    
    print(f"\n{'█' * 80}")
    print(f"  검사 완료: 총 {len(checkpoints)}개 체크포인트 분석됨")
    print(f"{'█' * 80}\n")
    
    return checkpoints


# ============================================================
# 단일 체크포인트 조회 (get_tuple 기반)
# ============================================================

async def inspect_single_checkpoint(
    checkpointer,
    config: dict,
) -> None:
    """
    특정 체크포인트 하나를 상세 조회합니다.
    
    Args:
        checkpointer: BaseCheckpointSaver 인스턴스
        config: 조회할 체크포인트의 config
                예: {"configurable": {"thread_id": "t1", "checkpoint_id": "xxx"}}
    """
    print(_format_separator("🔎 단일 체크포인트 상세 조회", "█"))
    
    checkpoint_tuple = await checkpointer.aget_tuple(config)
    
    if not checkpoint_tuple:
        print("  ❌ 해당 config로 체크포인트를 찾을 수 없습니다.")
        return
    
    print_config_info(checkpoint_tuple)
    print_metadata_info(checkpoint_tuple)
    print_checkpoint_info(checkpoint_tuple)
    print_pending_writes_info(checkpoint_tuple)
    print_parent_config_info(checkpoint_tuple)
    print_extra_attributes(checkpoint_tuple)