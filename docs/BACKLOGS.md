Backlogs

## 대기 중

### 1. 버전관리 시스템
- `is_validate` 열 추가
- 날짜 postfix로 최신 버전 입력 시 기존 버전 마크 처리
- 검색 시 유효 버전만 반환하도록 필터링 로직 구현

=> 혹은 최신 버전 기준으로?

### 2. 하이브리드 검색 (BM25 + Vector Search)

**현재 상태**: LightRAG는 벡터 검색과 LLM 기반 키워드 추출만 지원. BM25 미구현.

**구현 방안**:

| 방안 | 설명 | 복잡도 |
|------|------|--------|
| A. 하이브리드 레이어 추가 | `query.py`에 BM25 검색 추가 후 벡터 결과와 결합 | 중간 |
| B. 인덱스 동시 구축 | 문서 수집 시 BM25 인덱스도 함께 구축 | 높음 |
| C. 리랭킹 활용 | LightRAG의 기존 리랭킹 시스템에 BM25 스코어 통합 | 낮음 |

**결정 필요 사항**:
- 토크나이저: 영어(whitespace) vs 한국어(konlpy/mecab)
- 인덱스 저장: 메모리 vs 파일 영속화
- 점수 결합: 가중 평균 vs RRF(Reciprocal Rank Fusion)

**필요 의존성**:
```
rank-bm25>=0.2.2
konlpy>=0.6.0  # 한국어 지원 시
```

**참조 위치**:
- 벡터 검색: `lightrag/operate.py:3336` (`_get_vector_context`)
- 키워드 추출: `lightrag/operate.py:3195` (`get_keywords_from_query`)
- RAGAnything 쿼리: `raganything/query.py:100-392`