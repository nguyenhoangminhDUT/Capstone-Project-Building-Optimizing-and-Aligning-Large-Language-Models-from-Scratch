"""Two-tier Input/Output Safety Guardrails with block reason and latency tracking (ms)."""

import re
import time
from typing import Optional, Tuple

# ==============================================================================
# 1. CÁC MẪU REGEX NHẬN DIỆN PROMPT INJECTION & JAILBREAK
# ==============================================================================
PROMPT_INJECTION_PATTERNS = [
    r"ignore\s+(?:all\s+|any\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts|guidelines)",
    r"disregard\s+(?:all\s+)?(?:previous|prior)?\s*(?:instructions|guidelines|rules)",
    r"forget\s+(?:all\s+)?(?:previous|prior)?\s*(?:instructions|rules)",
    r"you\s+are\s+now\s+(?:in\s+)?dan\s+mode",
    r"\bdan\b\s+mode",
    r"do\s+anything\s+now",
    r"system\s*override",
    r"override\s+(?:safety|security|all\s+rules|filters)",
    r"system\s*prompt",
    r"bypass(?:\s+all)?\s+(?:safety|security|filters|guardrails|rules|restrictions|biometric)",
    r"jailbreak",
    r"developer\s*mode",
    r"unrestricted\s*mode",
    r"reveal\s+(?:your\s+)?(?:system\s+prompt|secret\s+key|internal\s+(?:instructions|configurations))",
    r"show\s+me\s+your\s+(?:system\s+prompt|internal\s+instructions)",
    r"new\s+instructions\s+follow",
    r"act\s+as\s+an\s+unfiltered",
]

PROMPT_INJECTION_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in PROMPT_INJECTION_PATTERNS),
    re.IGNORECASE,
)

# ==============================================================================
# 2. TỪ KHÓA & MẪU YÊU CẦU GÂY HẠI / VI PHẠM PHÁP LUẬT
# ==============================================================================
HARMFUL_REQUEST_PATTERNS = [
    # Bạo lực, vũ khí & chất độc
    r"\b(?:pipe\s+)?bombs?\b",
    r"\bexplosives?\b",
    r"\bweapons?\b",
    r"\bassassinate\b",
    r"\bmurder\b",
    r"\bmassacre\b",
    r"\bterrorist\b",
    r"\bsuicide\b",
    r"\bself-harm\b",
    r"\bpoison(?:\s+someone)?\b",
    # Ma túy & hàng cấm
    r"\bheroin\b",
    r"\bcocaine\b",
    r"\bmethamphetamine\b",
    r"\bfentanyl\b",
    r"\bsynthesize\s+(?:dangerous\s+)?(?:illegal\s+)?drugs\b",
    r"\billegal\s+drugs\b",
    r"\bsmuggle\s+.*contraband\b",
    # Tội phạm mạng & lừa đảo
    r"\bmalware\b",
    r"\bransomware\b",
    r"\bkeylogger\b",
    r"\bphishing\b",
    r"\bddos\b",
    r"\bexploit\s+script\b",
    r"\bzero-day\s+vulnerability\b",
    r"\bhack\s+into\b",
    r"\bsteal\s+(?:banking\s+)?passwords\b",
    r"\bsteal\s+credit\s+card\b",
    r"\bcredit\s+card\s+(?:theft|numbers\s+with\s+cvv)\b",
    r"\bhotwire\s+a\s+car\b",
    r"\bevade\s+police\b",
    r"\bburglary\b",
    # Lạm dụng, thù địch & xâm phạm riêng tư
    r"\bchild\s+abuse\b",
    r"\bhuman\s+trafficking\b",
    r"\bmoney\s+laundering\b",
    r"\blaunder\s+money\b",
    r"\bhate\s+speech\b",
    r"\bpersonal\s+phone\s+numbers\s+and\s+home\s+addresses\b",
]

HARMFUL_REQUEST_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in HARMFUL_REQUEST_PATTERNS),
    re.IGNORECASE,
)

# ==============================================================================
# 3. TẦNG LỌC OUTPUT (PII, HƯỚNG DẪN NGUY HIỂM, RÒ RỈ SYSTEM PROMPT)
# ==============================================================================
PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|\b0\d{9,10}\b"
)

EMAIL_REGEX = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)

SYSTEM_LEAKAGE_PATTERNS = [
    r"below\s+is\s+an\s+instruction\s+that\s+describes\s+a\s+task",
    r"my\s+system\s+prompt\s+is",
    r"system\s+prompt\s*:",
    r"internal\s+instructions\s*:",
    r"secret\s*key\s*:",
]

SYSTEM_LEAKAGE_REGEX = re.compile(
    "|".join(f"(?:{p})" for p in SYSTEM_LEAKAGE_PATTERNS),
    re.IGNORECASE,
)

OUTPUT_UNSAFE_PATTERNS = [
    r"kill\s+yourself",
    r"hate\s+speech",
    r"exterminate",
    r"subhuman",
    r"nigger",
    r"faggot",
    r"chink",
    r"gook",
    r"retard",
    r"death\s+to\s+all",
    r"how\s+to\s+make\s+a\s+(?:pipe\s+)?bomb",
    r"manufacture\s+explosives?",
    r"synthesize\s+(?:illegal\s+)?(?:drugs|narcotics)",
    r"steal\s+credit\s+card",
    r"write\s+(?:malware|ransomware|keylogger)",
    r"hack\s+into\s+government",
]

OUTPUT_UNSAFE_REGEX = re.compile(
    "|".join(rf"\b{p}\b" for p in OUTPUT_UNSAFE_PATTERNS),
    re.IGNORECASE,
)


def check_input(prompt: str) -> Tuple[bool, Optional[str], float]:
    """
    Tầng lọc đầu vào (Input Guardrail):
    Kiểm tra Prompt Injection, Jailbreak (DAN, ignore previous instructions) và yêu cầu gây hại/vi phạm pháp luật.

    Returns:
        (is_safe, block_reason, latency_ms)
    """
    t0 = time.perf_counter()
    clean_text = prompt.strip() if isinstance(prompt, str) else ""

    if not clean_text:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, "Empty prompt blocked by input guardrail.", latency_ms

    match_inj = PROMPT_INJECTION_REGEX.search(clean_text)
    if match_inj:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"Blocked by Input Guardrail: Prompt Injection / Jailbreak detected ('{match_inj.group(0)}').", latency_ms

    match_harm = HARMFUL_REQUEST_REGEX.search(clean_text)
    if match_harm:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"Blocked by Input Guardrail: Harmful / Illegal content detected ('{match_harm.group(0)}').", latency_ms

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return True, None, latency_ms


def check_output(response: str) -> Tuple[bool, Optional[str], float]:
    """
    Tầng lọc đầu ra (Output Guardrail):
    Kiểm tra nội dung độc hại, hướng dẫn nguy hiểm, PII (SĐT/Email) và rò rỉ system prompt.

    Returns:
        (is_safe, block_reason, latency_ms)
    """
    t0 = time.perf_counter()
    clean_text = response.strip() if isinstance(response, str) else ""

    if not clean_text:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return True, None, latency_ms

    match_leak = SYSTEM_LEAKAGE_REGEX.search(clean_text)
    if match_leak:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"Blocked by Output Guardrail: System prompt leakage detected ('{match_leak.group(0)}').", latency_ms

    if PHONE_REGEX.search(clean_text) or EMAIL_REGEX.search(clean_text):
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, "Blocked by Output Guardrail: PII (phone number or email) detected.", latency_ms

    match_unsafe = OUTPUT_UNSAFE_REGEX.search(clean_text)
    if match_unsafe:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return False, f"Blocked by Output Guardrail: Unsafe or toxic content detected ('{match_unsafe.group(0)}').", latency_ms

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return True, None, latency_ms


class SafetyGuardrails:
    """Lớp điều phối an toàn 2 tầng cho quá trình suy luận (Input & Output Guardrails)."""

    def __init__(self, max_repetition_threshold: int = 5) -> None:
        self.max_repetition_threshold = max_repetition_threshold

    def check_input(self, prompt: str) -> Tuple[bool, Optional[str]]:
        """Kiểm tra input prompt, trả về (is_safe, refusal_reason)."""
        is_safe, reason, _ = check_input(prompt)
        return is_safe, reason

    def check_input_with_latency(self, prompt: str) -> Tuple[bool, Optional[str], float]:
        """Kiểm tra input prompt, trả về (is_safe, refusal_reason, latency_ms)."""
        return check_input(prompt)

    def check_output(self, response_text: str) -> Tuple[bool, str]:
        """Kiểm tra output, lọc PII/độc hại/system leakage và xử lý vòng lặp từ vô hạn."""
        is_safe, reason, _ = check_output(response_text)
        if not is_safe:
            return False, reason or "Nội dung phản hồi bị chặn do vi phạm chính sách an toàn."

        words = response_text.split()
        if len(words) >= self.max_repetition_threshold:
            for i in range(len(words) - self.max_repetition_threshold + 1):
                window = words[i : i + self.max_repetition_threshold]
                if len(set(window)) == 1:
                    sanitized = " ".join(words[: i + 1]) + " [Phát hiện lặp từ, đã tự động ngắt kết quả]"
                    return True, sanitized

        return True, response_text

    def check_output_with_latency(self, response_text: str) -> Tuple[bool, str, float]:
        """Kiểm tra output, trả về (is_safe, sanitized_or_reason, latency_ms)."""
        is_safe, reason, latency_ms = check_output(response_text)
        if not is_safe:
            return False, reason or "Nội dung phản hồi bị chặn do vi phạm chính sách an toàn.", latency_ms
        return True, response_text, latency_ms
