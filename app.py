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


@st.cache_data(ttl=60)  # 60초마다 새로 가져오기
def fetch_players_from_sheet(default_score: int = 0) -> list[dict]:
    # 이름
    df_names = pd.read_csv(NAMES_URL, header=None)
    names = (
        df_names.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # 점수(음수 포함 허용)
    df_scores = pd.read_csv(SCORES_URL, header=None)
    scores_raw = df_scores.iloc[:, 0].tolist()

    scores = []
    for x in scores_raw:
        try:
            s = int(float(x))
        except Exception:
            s = default_score
        scores.append(s)

    # 길이 맞추기(이름 기준)
    if len(scores) < len(names):
        scores += [default_score] * (len(names) - len(scores))
    else:
        scores = scores[:len(names)]

    # (이름, 점수) 묶기 — 빈 이름 제거
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


# ----------------------------
# Team assignment (Method A: target-based)
# ----------------------------
def greedy_assign(players, team_count, team_size, seed=42):
    """
    음수 허용용 그리디 (부호 분리 + target 근접)
    - target = 전체합 / 팀수
    - |score| 큰 순으로 먼저 배치(영향 큰 값 분산)
    - score >= 0: '낮은 팀' 쪽을 우선으로 target 근접
    - score <  0: '높은 팀' 쪽을 우선으로 target 근접
    """
    rng = random.Random(seed)

    total = sum(float(p["score"]) for p in players)
    target = total / float(team_count)

    ordered = sorted(players, key=lambda p: abs(float(p["score"])), reverse=True)
    teams = [{"members": [], "sum": 0.0} for _ in range(team_count)]

    for p in ordered:
        s = float(p["score"])

        candidates = [i for i in range(team_count) if len(teams[i]["members"]) < team_size]
        if not candidates:
            break

        # 부호에 따른 "우선 후보군" 만들기
        if s >= 0:
            # 양수/0은 낮은 팀(합이 작은 쪽)에게 가야 평균화에 유리
            min_sum = min(teams[i]["sum"] for i in candidates)
            priority = [i for i in candidates if teams[i]["sum"] == min_sum]
        else:
            # 음수는 높은 팀(합이 큰 쪽)에게 가야 약팀 붕괴를 막음
            max_sum = max(teams[i]["sum"] for i in candidates)
            priority = [i for i in candidates if teams[i]["sum"] == max_sum]

        # 우선 후보군 안에서 target에 가장 가까워지는 팀을 선택
        best_idxs = []
        best_dist = None

        for i in priority:
            new_sum = teams[i]["sum"] + s
            dist = abs(new_sum - target)

            if best_dist is None or dist < best_dist - 1e-12:
                best_dist = dist
                best_idxs = [i]
            elif abs(dist - best_dist) <= 1e-12:
                best_idxs.append(i)

        # (안전장치) priority가 비어있을 수는 없지만, 혹시 모를 경우 candidates로 폴백
        if not best_idxs:
            best_idxs = candidates

        # 동률이면 인원 적은 팀 우선 -> 그래도 동률이면 랜덤
        if len(best_idxs) > 1:
            min_size = min(len(teams[i]["members"]) for i in best_idxs)
            best_idxs = [i for i in best_idxs if len(teams[i]["members"]) == min_size]

        chosen = rng.choice(best_idxs)
        teams[chosen]["members"].append(p)
        teams[chosen]["sum"] += s

    return teams

def recompute_team_sum(team: dict) -> None:
    team["sum"] = sum(float(m["score"]) for m in team["members"])

def move_member(teams: list, src_idx: int, member_id: int, dst_idx: int) -> None:
    """teams_result에서 src 팀의 member_id를 dst 팀으로 이동"""
    if src_idx == dst_idx:
        return

    src = teams[src_idx]
    dst = teams[dst_idx]

    # 찾기
    pos = None
    for i, m in enumerate(src["members"]):
        if m["id"] == member_id:
            pos = i
            break
    if pos is None:
        return

    member = src["members"].pop(pos)
    dst["members"].append(member)

    # 합계 재계산
    recompute_team_sum(src)
    recompute_team_sum(dst)


# ----------------------------
# UI
# ----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.subheader("선수 등록")

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

    st.divider()
    st.subheader(f"등록된 선수 ({len(st.session_state.players)}명)")
    st.caption("팀에 넣고 싶은 선수만 체크하세요.")

    if not st.session_state.players:
        st.caption("아직 등록된 선수가 없습니다.")
    else:
        # 전체 선택/해제
        btn1, btn2 = st.columns(2)

        with btn1:
            if st.button("전체 선택"):
                for p in st.session_state.players:
                    st.session_state[f"chk_{p['id']}"] = True
                st.rerun()

        with btn2:
            if st.button("전체 해제"):
                for p in st.session_state.players:
                    st.session_state[f"chk_{p['id']}"] = False
                st.rerun()

        st.write("")

        selected_ids = set()

        for idx, p in enumerate(st.session_state.players):
            key = f"chk_{p['id']}"
            if key not in st.session_state:
                st.session_state[key] = False

            c0, c1, c2 = st.columns([1.2, 6, 2])

            with c0:
                checked = st.checkbox("선택", key=key, label_visibility="collapsed")
                if checked:
                    selected_ids.add(p["id"])

            with c1:
                st.write(f"{idx + 1}. {p['name']}")

            with c2:
                st.write(f"점수: **{p['score']}**")

        st.session_state.selected_ids = selected_ids

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
        elif selected_count > required:
            st.error(f"선택된 선수가 초과입니다. {selected_count}명 / 필요 {required}명 (체크를 줄여주세요)")
            st.session_state.teams_result = None
        else:
            st.session_state.teams_result = greedy_assign(
                selected_players, int(team_count), TEAM_SIZE, seed=int(seed)
            )


# ----------------------------
# Results
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

    cols = st.columns(min(len(teams), 4))
    for i, t in enumerate(teams):
        with cols[i % len(cols)]:
            st.markdown(f"### 팀 {i + 1}")
            st.write(f"합계: **{t['sum']:.2f}** (target 대비: {t['sum'] - target:+.2f})")
            for m in t["members"]:
               teams = st.session_state.teams_result
if teams:
    cols = st.columns(min(len(teams), 4))

    for team_idx, t in enumerate(teams):
        with cols[team_idx % len(cols)]:
            st.markdown(f"### 팀 {team_idx + 1}")
            st.write(f"합계: **{t['sum']:.2f}**")

            # 멤버 표시 + 이동 UI
            for m in t["members"]:
                row1, row2 = st.columns([5, 3])

                with row1:
                    st.write(f"- {m['name']} (**{m['score']}**)")

                with row2:
                    # 목적지 팀 선택(자기 팀 제외)
                    options = [i for i in range(len(teams)) if i != team_idx]
                    label_map = {i: f"팀 {i+1}" for i in options}

                    dst = st.selectbox(
                        "이동 대상",
                        options=options,
                        format_func=lambda i: label_map[i],
                        key=f"mv_dst_{team_idx}_{m['id']}",
                        label_visibility="collapsed",
                    )

                    # 팀당 인원 5명 제한을 유지하고 싶으면 아래 체크 유지
                    disabled = (len(teams[dst]["members"]) >= TEAM_SIZE)

                    if st.button(
                        "이동",
                        key=f"mv_btn_{team_idx}_{m['id']}",
                        disabled=disabled,
                        use_container_width=True,
                    ):
                        move_member(teams, team_idx, m["id"], dst)
                        st.session_state.teams_result = teams
                        st.rerun()

