/**
 * ============================================================================
 * Deep-catch 프론트엔드 메인 컨트롤러 (app.js)
 * 담당: 송동화
 *
 * [핵심 설계 원칙]
 * 1. 외부 도메인(ngrok, Cloudflare Tunnel, 배포 서버 도메인 등) 접속 완벽 지원:
 *    - URL 쿼리 파라미터(?api=https://...), window.DEEP_CATCH_API_URL 전역 변수,
 *      localStorage 저장소, 동일 출처(Same-Origin) 순으로 백엔드 엔드포인트를 자동 탐색합니다.
 *    - 외부 도메인 간 통신(Cross-Origin)을 위해 fetch 옵션에 mode: "cors"를 명시합니다.
 * 2. 백엔드(database.py / ensemble_detector.py / FastAPI) 응답 구조 유연 호환:
 *    - prediction 필드가 "Fake" / "Real" 또는 "FAKE (가짜)" / "REAL (진짜)" 어느 형태로
 *      오더라도 정확히 판별하도록 정규화 파서(normalizePrediction)를 내장했습니다.
 * 3. 보안(XSS 방어):
 *    - DB에서 조회한 filename이나 외부 문자열을 innerHTML에 직접 삽입하지 않고
 *      DOM API(document.createElement, textContent)만 사용하여 XSS 공격을 원천 차단합니다.
 * ============================================================================
 */

"use strict";

/* ----------------------------------------------------------------------------
 * [1] 외부 도메인 대응 백엔드 API 주소(Base URL) 해석 및 관리 모듈
 * ---------------------------------------------------------------------------- */
const STORAGE_KEY_API_URL = "deepcatch_backend_api_url";
const DEFAULT_LOCAL_API = "http://localhost:8000";

/**
 * 현재 접속 환경(로컬 개발 / 외부 터널 도메인 / 리버스 프록시 배포 환경)에 맞춰
 * 가장 적절한 백엔드 API 기본 URL(끝 슬래시 제거)을 반환합니다.
 *
 * 우선순위:
 *  1) 접속 URL의 쿼리 스트링 (?api=https://example.ngrok-free.app) -> 공유 링크 접속 시 즉시 연동
 *  2) 브라우저 localStorage에 사용자가 직접 저장한 외부 백엔드 도메인
 *  3) HTML 전역 설정값 (window.DEEP_CATCH_API_URL)
 *  4) 현재 페이지가 http(s)로 서빙 중이고 로컬 파일(file://)이 아닌 경우:
 *     - 프론트와 백엔드가 같은 도메인/포트로 통합 배포되었다면 window.location.origin 사용
 *     - 로컬 개발 환경(localhost / 127.0.0.1)에서 프론트(5500 등)만 띄운 경우 http://localhost:8000 사용
 */
function resolveBackendBaseUrl() {
  // 1. URL 쿼리 파라미터(?api=...) 확인 (외부 사용자에게 링크 공유 시 매우 유용)
  const urlParams = new URLSearchParams(window.location.search);
  const queryApi = urlParams.get("api");
  if (queryApi && isValidHttpUrl(queryApi)) {
    const cleaned = sanitizeBaseUrl(queryApi);
    localStorage.setItem(STORAGE_KEY_API_URL, cleaned);
    return cleaned;
  }

  // 2. localStorage에 저장된 사용자 지정 외부 백엔드 도메인 확인
  const savedApi = localStorage.getItem(STORAGE_KEY_API_URL);
  if (savedApi && isValidHttpUrl(savedApi)) {
    return sanitizeBaseUrl(savedApi);
  }

  // 3. 전역 주입 변수 확인
  if (typeof window.DEEP_CATCH_API_URL === "string" && isValidHttpUrl(window.DEEP_CATCH_API_URL)) {
    return sanitizeBaseUrl(window.DEEP_CATCH_API_URL);
  }

  // 4. 기본 폴백(Fallback) 결정
  const { protocol, hostname, origin } = window.location;
  if (protocol === "file:") {
    return DEFAULT_LOCAL_API;
  }
  if (hostname === "localhost" || hostname === "127.0.0.1") {
    return DEFAULT_LOCAL_API;
  }
  // 외부 도메인(예: https://deep-catch.site)에서 Nginx/FastAPI 통합 서빙 시 동일 출처 사용
  return sanitizeBaseUrl(origin);
}

/** URL 문자열 유효성(http:// 또는 https://) 검증 */
function isValidHttpUrl(value) {
  try {
    const parsed = new URL(value.trim());
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

/** URL 끝의 불필요한 슬래시('/') 제거 */
function sanitizeBaseUrl(url) {
  return url.trim().replace(/\/+$/, "");
}

/** 현재 활성화된 백엔드 API Base URL 상태 */
let currentApiBaseUrl = resolveBackendBaseUrl();

/* ----------------------------------------------------------------------------
 * [2] DOM 요소 참조 캐싱 및 애플리케이션 상태 변수
 * ---------------------------------------------------------------------------- */
const dom = {
  // 외부 도메인 설정 및 연결 상태 UI
  serverStatusDot: document.getElementById("serverStatusDot"),
  serverStatusText: document.getElementById("serverStatusText"),
  toggleServerConfigBtn: document.getElementById("toggleServerConfigBtn"),
  serverConfigBox: document.getElementById("serverConfigBox"),
  apiBaseUrlInput: document.getElementById("apiBaseUrlInput"),
  saveApiUrlBtn: document.getElementById("saveApiUrlBtn"),
  resetApiUrlBtn: document.getElementById("resetApiUrlBtn"),

  // 파일 업로드 및 미리보기 UI
  dropZone: document.getElementById("dropZone"),
  imageInput: document.getElementById("imageInput"),
  uploadPlaceholder: document.getElementById("uploadPlaceholder"),
  previewWrapper: document.getElementById("previewWrapper"),
  imagePreview: document.getElementById("imagePreview"),
  selectedFileName: document.getElementById("selectedFileName"),
  selectedFileSize: document.getElementById("selectedFileSize"),
  clearFileBtn: document.getElementById("clearFileBtn"),
  predictBtn: document.getElementById("predictBtn"),
  uiFeedbackMsg: document.getElementById("uiFeedbackMsg"),

  // 분석 결과 시각화 UI
  resultEmptyState: document.getElementById("resultEmptyState"),
  resultContent: document.getElementById("resultContent"),
  verdictBanner: document.getElementById("verdictBanner"),
  verdictBadge: document.getElementById("verdictBadge"),
  verdictTitle: document.getElementById("verdictTitle"),
  verdictDesc: document.getElementById("verdictDesc"),
  ensembleProbValue: document.getElementById("ensembleProbValue"),
  analyzedFileLabel: document.getElementById("analyzedFileLabel"),
  ensembleProgressBar: document.getElementById("ensembleProgressBar"),

  // 3종 모델 세부 확률 UI
  spatialProbText: document.getElementById("spatialProbText"),
  spatialBar: document.getElementById("spatialBar"),
  freqProbText: document.getElementById("freqProbText"),
  freqBar: document.getElementById("freqBar"),
  dualProbText: document.getElementById("dualProbText"),
  dualBar: document.getElementById("dualBar"),

  // 탐지 이력 테이블 UI
  refreshLogsBtn: document.getElementById("refreshLogsBtn"),
  logTableBody: document.getElementById("logTableBody"),
};

/** 현재 선택된 이미지 파일 및 미리보기 ObjectURL 상태 */
let selectedFile = null;
let currentPreviewObjectUrl = null;
let isPredicting = false;

/* ----------------------------------------------------------------------------
 * [3] 공통 Fetch 유틸리티 (외부 도메인 CORS & 타임아웃 & 터널 헤더 대응)
 * ---------------------------------------------------------------------------- */
/**
 * 외부 도메인 백엔드(FastAPI)와 통신하기 위한 공통 fetch 래퍼 함수
 * - AbortController를 사용해 서버 무응답 시 무한 대기를 방지(timeoutMs)합니다.
 * - ngrok 무료 도메인 접속 시 브라우저 경고 HTML 페이지 대신 JSON 응답을 받기 위해
 *   'ngrok-skip-browser-warning' 헤더를 함께 전송합니다.
 */
async function apiFetch(endpointPath, options = {}, timeoutMs = 30000) {
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), timeoutMs);

  const fullUrl = `${currentApiBaseUrl}${endpointPath}`;
  const mergedHeaders = {
    "ngrok-skip-browser-warning": "true",
    ...(options.headers || {}),
  };

  try {
    const response = await fetch(fullUrl, {
      ...options,
      mode: "cors", // 외부 도메인 간 요청(Cross-Origin Resource Sharing) 명시
      headers: mergedHeaders,
      signal: controller.signal,
    });
    return response;
  } finally {
    clearTimeout(timerId);
  }
}

/* ----------------------------------------------------------------------------
 * [4] 백엔드 서버 연결 상태 확인 (Health Check) 및 외부 도메인 설정 UI 제어
 * ---------------------------------------------------------------------------- */
function updateServerStatusUI(state, message) {
  dom.serverStatusDot.className = `status-dot ${state}`;
  dom.serverStatusText.textContent = message;
  dom.serverStatusText.title = `${message} (${currentApiBaseUrl})`;
}

/**
 * 백엔드 API 서버의 가용성을 확인합니다.
 * 전용 헬스체크(/api/health)가 없더라도 /api/logs 응답 여부로 연결 상태를 판별합니다.
 */
async function checkBackendConnection() {
  updateServerStatusUI("checking", `연결 확인 중 (${currentApiBaseUrl})`);
  try {
    const res = await apiFetch("/api/logs", { method: "GET" }, 6000);
    if (res.ok) {
      updateServerStatusUI("online", `연결됨 · ${currentApiBaseUrl}`);
      const logs = await res.json();
      renderLogsTable(logs);
    } else {
      updateServerStatusUI("offline", `서버 응답 오류 (HTTP ${res.status})`);
    }
  } catch (err) {
    // HTTPS 페이지에서 HTTP 백엔드를 호출할 때 발생하는 Mixed Content 안내 포함
    const isMixedContentRisk =
      window.location.protocol === "https:" && currentApiBaseUrl.startsWith("http://");

    if (isMixedContentRisk) {
      updateServerStatusUI("offline", "차단됨: HTTPS 사이트에서는 HTTPS 백엔드 도메인 필요");
    } else {
      updateServerStatusUI("offline", `연결 실패 · ${currentApiBaseUrl}`);
    }
    renderLogsError("백엔드 서버에 연결할 수 없습니다. 우측 상단 [도메인 설정]에서 서버 주소를 확인하세요.");
  }
}

/* ----------------------------------------------------------------------------
 * [5] 이미지 파일 선택, 유효성 검증 및 실시간 미리보기 처리
 * ---------------------------------------------------------------------------- */
const ALLOWED_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB 제한

function setFeedbackMessage(text, isError = false) {
  dom.uiFeedbackMsg.textContent = text;
  dom.uiFeedbackMsg.classList.toggle("error", isError);
}

function handleFileSelect(file) {
  if (!file) return;

  // 1. 파일 확장자/MIME 타입 검증
  if (!ALLOWED_MIME_TYPES.includes(file.type)) {
    setFeedbackMessage("지원하지 않는 파일 형식입니다. JPG, PNG, WEBP 이미지만 업로드할 수 있습니다.", true);
    return;
  }

  // 2. 파일 용량 검증
  if (file.size > MAX_FILE_SIZE_BYTES) {
    setFeedbackMessage("파일 크기가 10MB를 초과했습니다. 더 작은 이미지를 선택해 주세요.", true);
    return;
  }

  // 기존 ObjectURL 메모리 해제 (메모리 누수 방지)
  if (currentPreviewObjectUrl) {
    URL.revokeObjectURL(currentPreviewObjectUrl);
  }

  selectedFile = file;
  currentPreviewObjectUrl = URL.createObjectURL(file);

  // 미리보기 화면 갱신
  dom.imagePreview.src = currentPreviewObjectUrl;
  dom.selectedFileName.textContent = file.name;
  dom.selectedFileSize.textContent = formatBytes(file.size);

  dom.uploadPlaceholder.classList.add("hidden");
  dom.previewWrapper.classList.remove("hidden");
  dom.predictBtn.disabled = false;
  setFeedbackMessage("이미지가 준비되었습니다. [딥페이크 앙상블 판별 실행] 버튼을 눌러주세요.");
}

function clearSelectedFile(event) {
  if (event) event.stopPropagation(); // dropZone 클릭 이벤트 전파 방지

  if (currentPreviewObjectUrl) {
    URL.revokeObjectURL(currentPreviewObjectUrl);
    currentPreviewObjectUrl = null;
  }

  selectedFile = null;
  dom.imageInput.value = "";
  dom.imagePreview.src = "";
  dom.previewWrapper.classList.add("hidden");
  dom.uploadPlaceholder.classList.remove("hidden");
  dom.predictBtn.disabled = true;
  setFeedbackMessage("");
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

/* ----------------------------------------------------------------------------
 * [6] 딥페이크 판별 API 요청 (POST /api/predict) 및 결과 렌더링
 * ---------------------------------------------------------------------------- */
/**
 * 백엔드(ensemble_detector.py 또는 FastAPI)에서 반환하는 다양한 판정 문자열을
 * 일관된 불리언(isFake) 및 한국어 라벨로 정규화합니다.
 */
function normalizePrediction(data) {
  const prob = Number(data.fake_probability ?? 0);
  const rawPred = String(data.prediction ?? "").toUpperCase();

  // is_fake 불리언 필드가 명시되어 있으면 우선 사용, 없으면 prediction 문자열 또는 50% 임계치 기준 판정
  const isFake =
    typeof data.is_fake === "boolean"
      ? data.is_fake
      : rawPred.includes("FAKE") || rawPred.includes("가짜") || prob >= 50.0;

  return {
    isFake,
    prob: Number.isFinite(prob) ? Math.min(Math.max(prob, 0), 100) : 0,
    filename: data.filename || (selectedFile ? selectedFile.name : "unknown_image"),
    spatialProb: data.spatial_probability ?? data.details?.spatial ?? null,
    freqProb: data.frequency_probability ?? data.details?.frequency ?? null,
    dualProb: data.dual_probability ?? data.details?.dual_stream ?? null,
  };
}

async function runDeepfakePrediction() {
  if (!selectedFile || isPredicting) return;

  isPredicting = true;
  dom.predictBtn.disabled = true;
  dom.predictBtn.textContent = "⏳ AI 앙상블 모델(Spatial · DCT · DualStream) 분석 중...";
  setFeedbackMessage("이미지를 서버로 전송하고 2D DCT 주파수 변환 및 추론을 수행 중입니다...");

  try {
    // multipart/form-data 형식으로 이미지 파일 포장 (Content-Type 헤더는 브라우저가 boundary 포함 자동 설정)
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await apiFetch("/api/predict", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      let detailMsg = `서버 오류 (HTTP ${response.status})`;
      try {
        const errJson = await response.json();
        if (errJson.detail) detailMsg = errJson.detail;
      } catch {
        // JSON 파싱 실패 시 기본 상태 코드 메시지 유지
      }
      throw new Error(detailMsg);
    }

    const resultJson = await response.json();
    renderPredictionResult(resultJson);
    setFeedbackMessage("분석이 완료되었으며 탐지 이력이 데이터베이스에 자동 저장되었습니다.");

    // 최신 탐지 이력 테이블 자동 갱신
    await fetchAndRenderLogs();
  } catch (error) {
    const isTimeout = error.name === "AbortError";
    const msg = isTimeout
      ? "분석 요청 시간이 초과되었습니다. 백엔드 GPU/CPU 부하 또는 네트워크 상태를 확인하세요."
      : `판별 실패: ${error.message}`;
    setFeedbackMessage(msg, true);
  } finally {
    isPredicting = false;
    dom.predictBtn.disabled = !selectedFile;
    dom.predictBtn.textContent = "🔍 딥페이크 앙상블 판별 실행";
  }
}

function renderPredictionResult(rawData) {
  const parsed = normalizePrediction(rawData);

  dom.resultEmptyState.classList.add("hidden");
  dom.resultContent.classList.remove("hidden");

  // 1. 판정 배너 스타일 및 문구 갱신
  dom.verdictBanner.classList.remove("result-fake", "result-real");
  dom.ensembleProgressBar.classList.remove("fake", "real");

  if (parsed.isFake) {
    dom.verdictBanner.classList.add("result-fake");
    dom.verdictBadge.textContent = "⚠️ DANGER · FAKE DETECTED";
    dom.verdictTitle.textContent = "FAKE (딥페이크 / AI 생성 이미지 의심)";
    dom.verdictDesc.textContent =
      "공간 픽셀 왜곡 또는 2D DCT 고주파 격자 노이즈에서 조작 아티팩트가 탐지되었습니다.";
    dom.ensembleProgressBar.classList.add("fake");
  } else {
    dom.verdictBanner.classList.add("result-real");
    dom.verdictBadge.textContent = "✅ SAFE · REAL IMAGE";
    dom.verdictTitle.textContent = "REAL (정상 원본 이미지 판정)";
    dom.verdictDesc.textContent =
      "공간 영역 및 주파수 영역 모두에서 뚜렷한 AI 생성·변조 흔적이 발견되지 않았습니다.";
    dom.ensembleProgressBar.classList.add("real");
  }

  // 2. 종합 확률 수치 및 게이지 바 애니메이션 반영
  dom.ensembleProbValue.textContent = `${parsed.prob.toFixed(2)}%`;
  dom.analyzedFileLabel.textContent = `파일: ${parsed.filename}`;
  dom.ensembleProgressBar.style.width = `${parsed.prob}%`;

  // 3. 3종 개별 모델 확률 표시 (백엔드 API가 세부 점수를 함께 내려줄 경우 자동 시각화)
  updateSubModelBar(dom.spatialProbText, dom.spatialBar, parsed.spatialProb);
  updateSubModelBar(dom.freqProbText, dom.freqBar, parsed.freqProb);
  updateSubModelBar(dom.dualProbText, dom.dualBar, parsed.dualProb);
}

function updateSubModelBar(textEl, barEl, value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    textEl.textContent = "통합 연산됨";
    barEl.style.width = "0%";
    return;
  }
  const num = Math.min(Math.max(Number(value), 0), 100);
  textEl.textContent = `${num.toFixed(2)}%`;
  barEl.style.width = `${num}%`;
}

/* ----------------------------------------------------------------------------
 * [7] SQLite 탐지 이력 조회 (GET /api/logs) 및 안전한 DOM 렌더링
 * ---------------------------------------------------------------------------- */
async function fetchAndRenderLogs() {
  try {
    const response = await apiFetch("/api/logs", { method: "GET" }, 8000);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const logs = await response.json();
    renderLogsTable(logs);
    updateServerStatusUI("online", `연결됨 · ${currentApiBaseUrl}`);
  } catch (error) {
    renderLogsError(`이력 조회 실패 (${error.message})`);
  }
}

function renderLogsTable(logs) {
  // 기존 행 초기화
  dom.logTableBody.innerHTML = "";

  if (!Array.isArray(logs) || logs.length === 0) {
    renderLogsError("아직 저장된 탐지 이력이 없습니다. 위에서 이미지를 업로드해 첫 분석을 실행해 보세요.");
    return;
  }

  // XSS 방지를 위해 innerHTML 대신 DOM 노드를 직접 생성하여 textContent로 바인딩
  logs.forEach((row) => {
    const tr = document.createElement("tr");

    const tdId = document.createElement("td");
    tdId.setAttribute("data-label", "ID");
    tdId.textContent = row.id ?? "-";

    const tdFilename = document.createElement("td");
    tdFilename.setAttribute("data-label", "검사 파일명");
    tdFilename.textContent = row.filename ?? "unknown";

    const tdPred = document.createElement("td");
    tdPred.setAttribute("data-label", "최종 판정");
    const badge = document.createElement("span");
    const predStr = String(row.prediction ?? "");
    const isFake = predStr.toUpperCase().includes("FAKE") || predStr.includes("가짜");
    badge.className = `pill ${isFake ? "pill-fake" : "pill-real"}`;
    badge.textContent = predStr || (isFake ? "FAKE (가짜)" : "REAL (진짜)");
    tdPred.appendChild(badge);

    const tdProb = document.createElement("td");
    tdProb.setAttribute("data-label", "가짜 확률");
    const probNum = Number(row.fake_probability);
    tdProb.textContent = Number.isFinite(probNum) ? `${probNum.toFixed(2)}%` : "-";

    const tdTime = document.createElement("td");
    tdTime.setAttribute("data-label", "탐지 일시");
    tdTime.textContent = row.created_at ?? "-";

    tr.append(tdId, tdFilename, tdPred, tdProb, tdTime);
    dom.logTableBody.appendChild(tr);
  });
}

function renderLogsError(message) {
  dom.logTableBody.innerHTML = "";
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.colSpan = 5;
  td.className = "table-empty";
  td.textContent = message;
  tr.appendChild(td);
  dom.logTableBody.appendChild(tr);
}

/* ----------------------------------------------------------------------------
 * [8] 이벤트 리스너 바인딩 및 초기 부트스트랩
 * ---------------------------------------------------------------------------- */
function bindEvents() {
  // 외부 도메인 설정 패널 토글
  dom.toggleServerConfigBtn.addEventListener("click", () => {
    dom.apiBaseUrlInput.value = currentApiBaseUrl;
    dom.serverConfigBox.classList.toggle("hidden");
  });

  // 외부 백엔드 도메인 URL 저장 및 즉시 재연결
  dom.saveApiUrlBtn.addEventListener("click", () => {
    const rawInput = dom.apiBaseUrlInput.value;
    if (!isValidHttpUrl(rawInput)) {
      alert("올바른 도메인 주소를 입력하세요. (예: https://api.deep-catch.site 또는 http://localhost:8000)");
      return;
    }
    currentApiBaseUrl = sanitizeBaseUrl(rawInput);
    localStorage.setItem(STORAGE_KEY_API_URL, currentApiBaseUrl);
    dom.serverConfigBox.classList.add("hidden");
    checkBackendConnection();
  });

  // 저장된 외부 도메인 초기화 (자동 감지 모드로 복귀)
  dom.resetApiUrlBtn.addEventListener("click", () => {
    localStorage.removeItem(STORAGE_KEY_API_URL);
    currentApiBaseUrl = resolveBackendBaseUrl();
    dom.apiBaseUrlInput.value = currentApiBaseUrl;
    dom.serverConfigBox.classList.add("hidden");
    checkBackendConnection();
  });

  // 드롭존 클릭 & 키보드(Enter/Space) 파일 선택기 열기
  dom.dropZone.addEventListener("click", () => dom.imageInput.click());
  dom.dropZone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      dom.imageInput.click();
    }
  });

  // 파일 입력 변경 감지
  dom.imageInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  });

  // 드래그 앤 드롭 이벤트
  dom.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dom.dropZone.classList.add("drag-over");
  });

  dom.dropZone.addEventListener("dragleave", () => {
    dom.dropZone.classList.remove("drag-over");
  });

  dom.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dom.dropZone.classList.remove("drag-over");
    if (e.dataTransfer?.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  });

  // 선택 파일 해제 및 판별 실행 버튼
  dom.clearFileBtn.addEventListener("click", clearSelectedFile);
  dom.predictBtn.addEventListener("click", runDeepfakePrediction);
  dom.refreshLogsBtn.addEventListener("click", fetchAndRenderLogs);
}

// 페이지 로드 완료 시 초기화 실행
document.addEventListener("DOMContentLoaded", () => {
  bindEvents();
  checkBackendConnection();
});
