import streamlit as st
import pandas as pd
import random

st.set_page_config(page_title="팀 밸런서", layout="wide")
st.title("팀 밸런서")

TEAM_SIZE = 5

# ----------------------------
# Session state init
# ----------------------------
SHEET_ID = "1raKWOAmdFv6tP51hW8JYjO6PHHvCtkKKIzFzqbppd3s"
GID = "1649695299"

# 각각 한 컬럼씩 가져오기 (B열=이름, N열=점수)
NAMES_URL  = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}&range=B7:B"
SCORES_URL = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/gviz/tq?tqx=out:csv&gid={GID}&range=O7:O"

@st.cache_data(ttl=60)  # 60초마다 새로고침(원하는 값으로 조절)
def fetch_players_from_sheet(default_score: int = 5) -> list[dict]:
    # 이름
    df_names = pd.read_csv(NAMES_URL, header=None)
    names = (
        df_names.iloc[:, 0]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    # 점수
    df_scores = pd.read_csv(SCORES_URL, header=None)
    scores_raw = df_scores.iloc[:, 0].tolist()

    # 점수 숫자화 + 정리
    scores = []
    for x in scores_raw:
        try:
            s = int(float(x))
        except Exception:
            s = default_score
        # 1~7 범위 보정
        if s < 0: s = 0
        if s > 7: s = 7
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



if "players" not in st.session_state:
    sheet_players = fetch_players_from_sheet(default_score=5)
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
# Greedy assignment
# ----------------------------
def greedy_assign(players, team_count, team_size, seed=42):
    rng = random.Random(seed)
    ordered = sorted(players, key=lambda p: p["score"], reverse=True)

    teams = [{"members": [], "sum": 0.0} for _ in range(team_count)]

    for p in ordered:
        candidates = [i for i in range(team_count) if len(teams[i]["members"]) < team_size]
        min_sum = min(teams[i]["sum"] for i in candidates)
        min_teams = [i for i in candidates if teams[i]["sum"] == min_sum]
        chosen = rng.choice(min_teams)

        teams[chosen]["members"].append(p)
        teams[chosen]["sum"] += float(p["score"])

    return teams


# ----------------------------
# UI
# ----------------------------
left, right = st.columns([1.35, 1])

with left:
    st.subheader("선수 등록")

    with st.form("add_player_form", clear_on_submit=True):
        name = st.text_input("선수 이름", placeholder="예: 긴꼬리딱새")
        score = st.selectbox("점수 (1~7)", options=list(range(1, 8)), index=3)
        submitted = st.form_submit_button("추가")

        if submitted:
            name = name.strip()
            if not name:
                st.warning("이름을 입력해 주세요.")
            else:
                # 중복 이름 막기(원하면 제거 가능)
                if any(p["name"] == name for p in st.session_state.players):
                    st.warning("이미 등록된 이름입니다.")
                else:
                    pid = st.session_state.next_id
                    st.session_state.next_id += 1
                    st.session_state.players.append({"id": pid, "name": name, "score": int(score)})
                    st.success(f"추가됨: {name} ({score})")

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
                    key = f"chk_{p['id']}"
                    st.session_state[key] = True
                st.rerun()

        with btn2:
            if st.button("전체 해제"):
                for p in st.session_state.players:
                    key = f"chk_{p['id']}"
                    st.session_state[key] = False
                st.rerun()

        st.write("")

        # 체크박스 상태로 selected_ids 재구성
        selected_ids = set()

        for idx, p in enumerate(st.session_state.players):
            key = f"chk_{p['id']}"

            # 최초 렌더링 시에만 기본값을 세션에 넣어둠
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
    st.info(f"팀 합계 최대-최소 차이(gap): **{gap:.2f}**")

    cols = st.columns(min(len(teams), 4))
    for i, t in enumerate(teams):
        with cols[i % len(cols)]:
            st.markdown(f"### 팀 {i + 1}")
            st.write(f"합계: **{t['sum']:.2f}**")
            for m in t["members"]:
                st.write(f"- {m['name']} (**{m['score']}**)")


