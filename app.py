import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="팀 밸런서", layout="wide")
st.title("팀 밸런서")

TEAM_SIZE = 5

# ----------------------------
# Google Sheets config
# ----------------------------
SHEET_ID = "1raKWOAmdFv6tP51hW8JYjO6PHHvCtkKKIzFzqbppd3s"
GID = "1649695299"

NAMES_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}&range=B7:B"
SCORES_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}&range=N7:N"


@st.cache_data(ttl=60)
def fetch_players_from_sheet(default_score: int = 0) -> list[dict]:
    df_names = pd.read_csv(NAMES_URL, header=None)
    names = (
        df_names.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    df_scores = pd.read_csv(SCORES_URL, header=None)
    scores_raw = df_scores.iloc[:, 0].tolist()

    scores = []
    for x in scores_raw:
        try:
            s = int(float(x))
        except Exception:
            s = default_score
        scores.append(s)

    if len(scores) < len(names):
        scores += [default_score] * (len(names) - len(scores))
    else:
        scores = scores[:len(names)]

    players = []
    for i, name in enumerate(names):
        if name:
            players.append({"name": name, "score": scores[i]})

    return players


# ----------------------------
# Session state init
# ----------------------------
if "players" not in st.session_state:
    sheet_players = fetch_players_from_sheet(default_score=0)
    st.session_state.players = [
        {"id": i + 1, "name": p["name"], "score": int(p["score"])}
        for i, p in enumerate(sheet_players)
    ]

if "next_id" not in st.session_state:
    st.session_state.next_id = len(st.session_state.players) + 1

if "selected_ids" not in st.session_state:
    st.session_state.selected_ids = set()

if "teams_result" not in st.session_state:
    st.session_state.teams_result = None

if "swap_pick" not in st.session_state:
    st.session_state.swap_pick = None  # (team_idx, member_id)


# ----------------------------
# Team assignment (부호 분리 + target 근접)
# ----------------------------
def greedy_assign(players, team_count, team_size, seed=42):
    """
    시드에 따라 결과가 눈에 띄게 바뀌는 버전:
    1) 시드로 입력 순서를 셔플 (같은 점수 그룹/동점 상황을 크게 흔듦)
    2) 팀 선택 점수(dist)에 아주 작은 노이즈(jitter)를 더해
       거의 비슷한 후보들 사이에서 랜덤 분기 발생
    3) 음수/양수 분리 규칙은 유지
    """
    rng = random.Random(seed)

    total = sum(float(p["score"]) for p in players)
    target = total / float(team_count)

    # ✅ 1) 시드로 먼저 셔플 → 시드 바꾸면 결과가 확 달라짐
    ordered = players[:]
    rng.shuffle(ordered)

    # 영향 큰 순으로(절대값 큰 점수 먼저)
    ordered.sort(key=lambda p: abs(float(p["score"])), reverse=True)

    teams = [{"members": [], "sum": 0.0} for _ in range(team_count)]

    # 노이즈 크기(점수 스케일에 맞춰 자동 설정)
    # 점수들이 작으면 노이즈도 작게, 점수 스케일이 크면 조금 키움
    abs_scores = [abs(float(p["score"])) for p in players]
    scale = max(abs_scores) if abs_scores else 1.0
    jitter = max(1e-6, scale * 0.02)  # 2% 정도 (원하면 0.01~0.05로 조절)

    for p in ordered:
        s = float(p["score"])

        candidates = [i for i in range(team_count) if len(teams[i]["members"]) < team_size]
        if not candidates:
            break

        # ✅ 음수/양수 분리 우선 후보군
        if s >= 0:
            min_sum = min(teams[i]["sum"] for i in candidates)
            priority = [i for i in candidates if teams[i]["sum"] == min_sum]
        else:
            max_sum = max(teams[i]["sum"] for i in candidates)
            priority = [i for i in candidates if teams[i]["sum"] == max_sum]

        # ✅ 2) target 근접(dist)에 작은 랜덤 노이즈를 섞어서 시드 영향 확대
        best_val = None
        best_idxs = []

        for i in priority:
            new_sum = teams[i]["sum"] + s
            dist = abs(new_sum - target)

            # dist가 거의 같을 때만 갈리도록 "아주 작은" 노이즈 추가
            noisy = dist + rng.uniform(-jitter, jitter)

            if best_val is None or noisy < best_val:
                best_val = noisy
                best_idxs = [i]
            elif abs(noisy - best_val) <= 1e-12:
                best_idxs.append(i)

        if not best_idxs:
            best_idxs = priority if priority else candidates

        # 동률이면 인원 적은 팀 우선 -> 그래도 동률이면 랜덤
        if len(best_idxs) > 1:
            min_size = min(len(teams[i]["members"]) for i in best_idxs)
            best_idxs = [i for i in best_idxs if len(teams[i]["members"]) == min_size]

        chosen = rng.choice(best_idxs)
        teams[chosen]["members"].append(p)
        teams[chosen]["sum"] += s

    return teams



# ----------------------------
# Selection handling (검색해도 선택 안 풀리게)
# ----------------------------
def on_toggle_player(pid: int, cb_key: str):
    if st.session_state.get(cb_key, False):
        st.session_state.selected_ids.add(pid)
    else:
        st.session_state.selected_ids.discard(pid)


# ----------------------------
# Swap helpers
# ----------------------------
def recompute_team_sum(team: dict) -> None:
    team["sum"] = sum(float(m["score"]) for m in team["members"])


def swap_members(teams: list, a: tuple[int, int], b: tuple[int, int]) -> None:
    ta, ida = a
    tb, idb = b

    if ta == tb and ida == idb:
        return

    pa = None
    pb = None

    for i, m in enumerate(teams[ta]["members"]):
        if m["id"] == ida:
            pa = i
            break

    for i, m in enumerate(teams[tb]["members"]):
        if m["id"] == idb:
            pb = i
            break

    if pa is None or pb is None:
        return

    teams[ta]["members"][pa], teams[tb]["members"][pb] = (
        teams[tb]["members"][pb],
        teams[ta]["members"][pa],
    )

    recompute_team_sum(teams[ta])
    recompute_team_sum(teams[tb])


# ----------------------------
# UI layout
# ----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.subheader("선수 등록")

    # 추가 폼
    with st.form("add_player_form", clear_on_submit=True):
        name = st.text_input("선수 이름", placeholder="예: 긴꼬리딱새")
        score = st.number_input("점수 (음수 가능)", value=0, step=1)
        submitted = st.form_submit_button("추가")

        if submitted:
            name = name.strip()
            if not name:
                st.warning("이름을 입력해 주세요.")
            else:
                if any(p["name"] == name for p in st.session_state.players):
                    st.warning("이미 등록된 이름입니다.")
                else:
                    pid = st.session_state.next_id
                    st.session_state.next_id += 1
                    st.session_state.players.append({"id": pid, "name": name, "score": int(score)})
                    st.success(f"추가됨: {name} ({int(score)})")
                    st.rerun()

    st.divider()
    st.subheader(f"등록된 선수 ({len(st.session_state.players)}명)")
    st.caption("팀에 넣고 싶은 선수만 체크하세요.")

    if not st.session_state.players:
        st.caption("아직 등록된 선수가 없습니다.")
    else:
        # 🔎 검색
        query = st.text_input("선수 검색", value="", placeholder="이름을 입력하면 필터링됩니다 (예: 딱새)")
        q = query.strip().lower()
        if q:
            visible_players = [p for p in st.session_state.players if q in p["name"].lower()]
        else:
            visible_players = st.session_state.players

        st.caption(f"표시 중: {len(visible_players)}명 / 전체: {len(st.session_state.players)}명")

        # 전체 선택/해제 (검색 결과에만 적용)
        btn1, btn2 = st.columns(2)
        with btn1:
            if st.button("전체 선택"):
                for p in visible_players:
                    pid = p["id"]
                    st.session_state.selected_ids.add(pid)
                    cb_key = f"cb_{pid}"
                    st.session_state[cb_key] = True
                st.rerun()

        with btn2:
            if st.button("전체 해제"):
                for p in visible_players:
                    pid = p["id"]
                    st.session_state.selected_ids.discard(pid)
                    cb_key = f"cb_{pid}"
                    st.session_state[cb_key] = False
                st.rerun()

        st.write("")

        # 체크 UI (선택의 진짜 저장소 = selected_ids)
        for idx, p in enumerate(visible_players):
            pid = p["id"]
            cb_key = f"cb_{pid}"

            c0, c1, c2 = st.columns([1.2, 6, 2])
            with c0:
                st.checkbox(
                    "선택",
                    value=(pid in st.session_state.selected_ids),
                    key=cb_key,
                    label_visibility="collapsed",
                    on_change=on_toggle_player,
                    args=(pid, cb_key),
                )
            with c1:
                st.write(f"{idx + 1}. {p['name']}")
            with c2:
                st.write(f"점수: **{p['score']}**")


with right:
    st.subheader("팀 설정")

    team_count = st.slider("팀 수", min_value=2, max_value=8, value=2, step=1)
    st.write(f"팀당 인원수: **{TEAM_SIZE}명**")

    required = team_count * TEAM_SIZE
    selected_count = len(st.session_state.selected_ids)

    st.write(f"필요 인원: **{required}명**")
    st.write(f"선택된 인원: **{selected_count}명**")

    seed = st.number_input("동률 랜덤 시드", min_value=0, max_value=10_000_000, value=1, step=1)

    st.divider()

    if st.button("선택된 선수로 팀 배정", type="primary", use_container_width=True):
        selected_players = [p for p in st.session_state.players if p["id"] in st.session_state.selected_ids]

        if selected_count < required:
            st.error(f"선택된 선수가 부족합니다. {selected_count}명 / 필요 {required}명")
            st.session_state.teams_result = None
            st.session_state.swap_pick = None
        elif selected_count > required:
            st.error(f"선택된 선수가 초과입니다. {selected_count}명 / 필요 {required}명 (체크를 줄여주세요)")
            st.session_state.teams_result = None
            st.session_state.swap_pick = None
        else:
            st.session_state.teams_result = greedy_assign(
                selected_players, int(team_count), TEAM_SIZE, seed=int(seed)
            )
            st.session_state.swap_pick = None
            st.rerun()


# ----------------------------
# Results (두 번 클릭하면 자동 스왑)
# ----------------------------
st.divider()
st.subheader("팀 배정 결과")

teams = st.session_state.teams_result
if not teams:
    st.caption("아직 팀 배정을 실행하지 않았습니다.")
else:
    sums = [t["sum"] for t in teams]
    gap = max(sums) - min(sums)

    total = sum(sums)
    target = total / len(sums)

    st.info(
        f"팀 합계 최대-최소 차이(gap): **{gap:.2f}**  |  "
        f"목표 평균(target): **{target:.2f}**"
    )

    st.caption("스왑: 사람 1명 클릭 → 다른 사람 1명 클릭하면 즉시 서로 교환됩니다. (같은 사람 다시 클릭하면 선택 해제)")

    cols = st.columns(min(len(teams), 4))
    for team_idx, t in enumerate(teams):
        with cols[team_idx % len(cols)]:
            st.markdown(f"### 팀 {team_idx + 1}")
            st.write(f"합계: **{t['sum']:.2f}** (target 대비: {t['sum'] - target:+.2f})")

            for m in t["members"]:
                picked = st.session_state.swap_pick
                is_picked = (picked == (team_idx, m["id"]))
                label = f"{'✅ ' if is_picked else ''}{m['name']} ({m['score']})"

                if st.button(label, key=f"pick_{team_idx}_{m['id']}", use_container_width=True):
                    if st.session_state.swap_pick is None:
                        st.session_state.swap_pick = (team_idx, m["id"])
                        st.rerun()
                    elif st.session_state.swap_pick == (team_idx, m["id"]):
                        st.session_state.swap_pick = None
                        st.rerun()
                    else:
                        a = st.session_state.swap_pick
                        b = (team_idx, m["id"])
                        swap_members(teams, a, b)
                        st.session_state.teams_result = teams
                        st.session_state.swap_pick = None
                        st.rerun()


