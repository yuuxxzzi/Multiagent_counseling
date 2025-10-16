from __future__ import annotations
from typing import List, Dict, Any, Optional, Tuple
import os, json, glob, math


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


class TemplateRAG:
    """`role_playing_templates/*.json` 내 역할극 템플릿을 로드하고
    임베딩 기반 검색을 제공한다. 카테고리 A/C/D 파일이 추가되어도 자동 인덱싱된다.
    """

    def __init__(self, templates_dir: str = "role_playing_templates", client=None, embedding_model: str = "text-embedding-3-small") -> None:
        self.templates_dir = templates_dir
        self.client = client
        self.embedding_model = embedding_model
        self.items: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self._load_templates()
        self._build_embeddings()

    def _iter_template_files(self) -> List[str]:
        pattern = os.path.join(self.templates_dir, "*.json")
        return sorted(glob.glob(pattern))

    def _load_templates(self) -> None:
        self.items = []
        for path in self._iter_template_files():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for it in data:
                        if isinstance(it, dict):
                            it = dict(it)
                            it["__file__"] = os.path.basename(path)
                            self.items.append(it)
            except Exception:
                continue

    def _template_text(self, it: Dict[str, Any]) -> str:
        parts: List[str] = []
        parts.append(str(it.get("id", "")))
        parts.append(str(it.get("title", "")))
        parts.append(str(it.get("scene_setup", "")))
        roles = it.get("roles", {}) or {}
        parts.append(str(roles.get("rp_agent_role", "")))
        parts.append(str(roles.get("user_role", "")))
        pre = it.get("preconditions", {}) or {}
        et = pre.get("emotion_tags", []) or []
        tt = pre.get("topic_tags", []) or []
        parts.extend([str(x) for x in et])
        parts.extend([str(x) for x in tt])
        for obj in it.get("objectives", []) or []:
            parts.append(str(obj))
        for con in it.get("constraints", []) or []:
            parts.append(str(con))
        for df in it.get("dialogue_flow", []) or []:
            if isinstance(df, dict):
                parts.append(str(df.get("stage", "")))
                parts.append(str(df.get("coach", "")))
                parts.append(str(df.get("rp", "")))
        return " \n ".join([p for p in parts if p])

    def _build_embeddings(self) -> None:
        self.embeddings = []
        if not self.items:
            return
        if self.client is None:
            return
        texts = [self._template_text(it) for it in self.items]
        batch_size = 64
        out: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            try:
                resp = self.client.embeddings.create(model=self.embedding_model, input=chunk)
                out.extend([d.embedding for d in resp.data])
            except Exception:
                dim = 1536
                out.extend([[0.0] * dim for _ in chunk])
        self.embeddings = out

    def _embed_query(self, query: str) -> List[float]:
        if self.client is None:
            return [0.0] * (len(self.embeddings[0]) if self.embeddings else 0)
        try:
            e = self.client.embeddings.create(model=self.embedding_model, input=[query])
            return e.data[0].embedding
        except Exception:
            return [0.0] * (len(self.embeddings[0]) if self.embeddings else 0)

    def search(self, query: str, k: int = 3, topic_hint: Optional[str] = None) -> List[Tuple[Dict[str, Any], float]]:
        if not self.items or not self.embeddings:
            return []
        qv = self._embed_query(query)
        scores: List[Tuple[int, float]] = []
        for idx, emb in enumerate(self.embeddings):
            s = _cosine_similarity(qv, emb)
            if topic_hint:
                it = self.items[idx]
                text = (it.get("title", "") + " " + it.get("id", "") + " " + it.get("scene_setup", "")).lower()
                pre = it.get("preconditions", {}) or {}
                tags = " ".join((pre.get("topic_tags", []) or [])).lower()
                if topic_hint.lower() in text or topic_hint.lower() in tags:
                    s += 0.15  # 부스트 강화 (면접 등 명시 키워드 가중)
            # 간단 키워드 매칭 추가 부스트 (query에 포함된 핵심어)
            it = self.items[idx]
            title_lower = (it.get("title", "") + " " + it.get("id", "") + " " + it.get("scene_setup", "")).lower()
            query_lower = query.lower()
            
            # 갈등 관련 키워드들
            conflict_keywords = ["갈등", "conflict", "싸움", "다툼", "불화", "문제", "친구"]
            for kw in conflict_keywords:
                if kw in query_lower and kw in title_lower:
                    s += 0.2  # 갈등 관련은 더 높은 가중치
                    break
            
            # 기타 키워드들
            other_keywords = ["면접", "interview", "발표", "presentation", "자신감", "스피치"]
            for kw in other_keywords:
                if kw in query_lower and kw in title_lower:
                    s += 0.1
                    break
            scores.append((idx, s))
        scores.sort(key=lambda x: x[1], reverse=True)
        out: List[Tuple[Dict[str, Any], float]] = []
        for idx, sc in scores[:max(1, k)]:
            out.append((self.items[idx], float(sc)))
        return out

    def best_template_for(self, user_text: str, emotion: Dict[str, Any], state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        topic = state.get("roleplay_topic") or ""
        q = f"사용자 발화: {user_text}\n의도 주제: {topic}\n감정: {emotion.get('emotion_class', '')}({emotion.get('emotion_score', 0.0)})"
        results = self.search(q, k=3, topic_hint=topic or None)
        return results[0][0] if results else None


