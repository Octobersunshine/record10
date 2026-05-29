import re
from typing import Any, Dict, List, Union


DEFAULT_SENSITIVE_FIELDS = [
    "password",
    "pwd",
    "secret",
    "token",
    "authorization",
    "auth",
    "credit_card",
    "card_number",
    "cvv",
    "ssn",
    "id_card",
    "phone",
    "mobile",
    "email",
]


class SensitiveDataMasker:
    def __init__(
        self,
        sensitive_fields: List[str] = None,
        mask_char: str = "*",
        mask_length: int = 8,
        preserve_length: bool = False,
    ):
        self.sensitive_fields = set(
            field.lower() for field in (sensitive_fields or DEFAULT_SENSITIVE_FIELDS)
        )
        self.mask_char = mask_char
        self.mask_length = mask_length
        self.preserve_length = preserve_length

    def _is_sensitive(self, key: str) -> bool:
        key_lower = key.lower()
        return key_lower in self.sensitive_fields

    def _mask_value(self, value: Any) -> Any:
        if value is None:
            return None

        if isinstance(value, str):
            if self.preserve_length:
                return self.mask_char * len(value)
            return self.mask_char * self.mask_length

        if isinstance(value, (int, float)):
            return self.mask_char * self.mask_length

        if isinstance(value, bool):
            return value

        return value

    def mask(
        self, data: Union[Dict[str, Any], List[Any], Any]
    ) -> Union[Dict[str, Any], List[Any], Any]:
        if data is None:
            return None

        if isinstance(data, dict):
            return self._mask_dict(data)

        if isinstance(data, list):
            return [self.mask(item) for item in data]

        return data

    def _mask_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        result = {}
        for key, value in data.items():
            if self._is_sensitive(key):
                result[key] = self._mask_value(value)
            elif isinstance(value, dict):
                result[key] = self._mask_dict(value)
            elif isinstance(value, list):
                result[key] = [self.mask(item) for item in value]
            else:
                result[key] = value
        return result


def mask_sensitive_data(
    data: Union[Dict[str, Any], List[Any], Any],
    sensitive_fields: List[str] = None,
) -> Union[Dict[str, Any], List[Any], Any]:
    masker = SensitiveDataMasker(sensitive_fields=sensitive_fields)
    return masker.mask(data)


def mask_email(email: str) -> str:
    if not email or "@" not in email:
        return email
    username, domain = email.split("@", 1)
    if len(username) <= 2:
        masked_username = username[0] + "***"
    else:
        masked_username = username[0] + "***" + username[-1]
    return f"{masked_username}@{domain}"


def mask_phone(phone: str) -> str:
    if not phone:
        return phone
    digits = re.sub(r"\D", "", phone)
    if len(digits) < 7:
        return phone
    return digits[:3] + "****" + digits[-4:]


def mask_ip(ip: str, anonymize: bool = False) -> str:
    if not ip or not anonymize:
        return ip
    parts = ip.split(".")
    if len(parts) == 4:
        return f"{parts[0]}.{parts[1]}.***.***"
    return ip
