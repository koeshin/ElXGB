---
description: ELXGB 단계별 구현 싸이클 (Zanjanbari Workflow)
---

# Zanjanbari Implementation Workflow

사용자가 "구현 진행해"라고 승인하면 즉각적으로 아래 6단계 워크플로우를 따라 하나의 모듈 단위 테스트를 완성해야 합니다.

1. **상태 파악 (State Check)**: `docs/implementation_plan.md`와 `docs/task.md`를 열어 이번 턴에 구현할 가장 우선적인 단일 태스크 1개를 선택합니다.
2. **논문 분석 에이전트 브리핑**: 논문 내의 어떤 Section/Algorithm인지, 주의할 보안성 제약은 무엇인지, 이번 코딩의 정확한 허용한도(커트라인)를 브리핑합니다.
3. **구현 에이전트 코딩**: 커트라인 내에서만 코드를 작성(`data_aligner.py`, `heservice.py` 등)하고, 단일 단위 코드를 검증하는 테스트 스크립트(`test_x.py`)를 함께 만듭니다.
4. **테스트 (Testing) 실행**: `python test_x.py`를 즉각 실행(`run_command`)하여 터미널 결과에서 통과(Pass)를 확인합니다.
5. **확인 에이전트 검증**: 테스트가 통과된 코드를 보고, 논문의 수식과 보안 법칙(예: HMAC 필수 사용, 평문 통신 불가 등)이 준수되었는지 감찰합니다. 
6. **플랜 문서 동기화**: 모든 검증이 끝나면 `docs/implementation_plan.md` 내 해당 모듈/아키텍처 항목 위치에 `~~취소선~~`을 그어 "설계도면에서 구체화 완료" 표시를 남기고, Task 리스트를 체크한 뒤 알립니다.
