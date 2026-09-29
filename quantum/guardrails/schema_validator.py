"""
Local dataset schema validation with privacy preservation.
Validates Alpaca, ChatML, ShareGPT formats without leaking text content.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class DatasetValidationError(Exception):
    pass


class SchemaValidator:
    SUPPORTED_SCHEMAS = ("alpaca", "chatml", "sharegpt", "auto")

    @staticmethod
    def detect_schema(record: Dict[str, Any]) -> str:
        """Infer format from record keys."""
        if "instruction" in record and "output" in record:
            return "alpaca"
        if "messages" in record and isinstance(record["messages"], list):
            return "chatml"
        if "conversations" in record and isinstance(record["conversations"], list):
            return "sharegpt"
        return "unknown"

    @classmethod
    def validate_record(cls, record: Any, schema: str, line_no: int) -> Tuple[bool, Optional[str]]:
        """
        Validates a single parsed record.
        IMPORTANT: Error strings must NOT contain record text or PII.
        """
        if not isinstance(record, dict):
            return False, f"Linha {line_no}: o registro não é um objeto JSON válido (tipo {type(record).__name__})."

        actual_schema = cls.detect_schema(record) if schema == "auto" else schema

        if actual_schema == "alpaca":
            instr = record.get("instruction")
            out = record.get("output")
            if not isinstance(instr, str) or not instr.strip():
                return False, f"Linha {line_no}: campo 'instruction' ausente ou vazio."
            if not isinstance(out, str) or not out.strip():
                return False, f"Linha {line_no}: campo 'output' ausente ou vazio."
            return True, None

        elif actual_schema == "chatml":
            msgs = record.get("messages")
            if not isinstance(msgs, list) or len(msgs) < 2:
                return False, f"Linha {line_no}: campo 'messages' deve ser uma lista com pelo menos 2 mensagens."
            for idx, msg in enumerate(msgs):
                if not isinstance(msg, dict):
                    return False, f"Linha {line_no}: mensagem #{idx+1} não é um objeto."
                role = msg.get("role")
                content = msg.get("content")
                if role not in ("system", "user", "assistant", "tool"):
                    return False, f"Linha {line_no}: role inválido '{role}' na mensagem #{idx+1}."
                if not isinstance(content, str) or not content.strip():
                    return False, f"Linha {line_no}: content vazio ou inválido na mensagem #{idx+1}."
            return True, None

        elif actual_schema == "sharegpt":
            convs = record.get("conversations")
            if not isinstance(convs, list) or len(convs) < 2:
                return False, f"Linha {line_no}: campo 'conversations' deve ser uma lista com pelo menos 2 turnos."
            for idx, turn in enumerate(convs):
                if not isinstance(turn, dict):
                    return False, f"Linha {line_no}: turno #{idx+1} não é um objeto."
                from_role = turn.get("from")
                val = turn.get("value")
                if from_role not in ("human", "gpt", "system", "chatgpt", "bing"):
                    return False, f"Linha {line_no}: from inválido '{from_role}' no turno #{idx+1}."
                if not isinstance(val, str) or not val.strip():
                    return False, f"Linha {line_no}: value vazio ou inválido no turno #{idx+1}."
            return True, None

        return False, f"Linha {line_no}: schema '{actual_schema}' não reconhecido."

    @classmethod
    def extract_full_text(cls, record: Dict[str, Any], schema: str) -> str:
        """Extract plain concatenated text for local token counting only."""
        actual_schema = cls.detect_schema(record) if schema == "auto" else schema
        if actual_schema == "alpaca":
            instr = record.get("instruction", "")
            inp = record.get("input", "")
            out = record.get("output", "")
            return f"{instr}\n{inp}\n{out}"
        elif actual_schema == "chatml":
            msgs = record.get("messages", [])
            return "\n".join(str(m.get("content", "")) for m in msgs if isinstance(m, dict))
        elif actual_schema == "sharegpt":
            convs = record.get("conversations", [])
            return "\n".join(str(c.get("value", "")) for c in convs if isinstance(c, dict))
        return " ".join(str(v) for v in record.values())

    @classmethod
    def validate_file(
        cls,
        file_path: Path | str,
        schema: str = "auto",
        max_errors: int = 15,
    ) -> Dict[str, Any]:
        """
        Validates a dataset file (.jsonl, .json, or .csv) locally.
        Returns aggregate stats and sanitized error messages.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")

        total_records = 0
        valid_records = 0
        invalid_records = 0
        errors: List[str] = []
        detected_schema = schema
        sample_texts: List[str] = []

        suffix = path.suffix.lower()

        if suffix in (".jsonl", ".txt"):
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    total_records += 1
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        invalid_records += 1
                        if len(errors) < max_errors:
                            errors.append(f"Linha {line_no}: Sintaxe JSON inválida.")
                        continue

                    if detected_schema == "auto" and total_records == 1:
                        detected_schema = cls.detect_schema(record)

                    is_valid, err = cls.validate_record(record, detected_schema, line_no)
                    if is_valid:
                        valid_records += 1
                        sample_texts.append(cls.extract_full_text(record, detected_schema))
                    else:
                        invalid_records += 1
                        if len(errors) < max_errors and err:
                            errors.append(err)

        elif suffix == ".json":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    raise DatasetValidationError(f"Arquivo {path.name} contém JSON corrompido.")

            if not isinstance(data, list):
                raise DatasetValidationError(f"Arquivo {path.name} deve conter uma lista JSON no nível raiz.")

            total_records = len(data)
            if detected_schema == "auto" and total_records > 0 and isinstance(data[0], dict):
                detected_schema = cls.detect_schema(data[0])

            for idx, record in enumerate(data, start=1):
                is_valid, err = cls.validate_record(record, detected_schema, idx)
                if is_valid:
                    valid_records += 1
                    sample_texts.append(cls.extract_full_text(record, detected_schema))
                else:
                    invalid_records += 1
                    if len(errors) < max_errors and err:
                        errors.append(err)

        elif suffix == ".csv":
            detected_schema = "alpaca"
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                if "instruction" not in fieldnames or "output" not in fieldnames:
                    raise DatasetValidationError(
                        f"CSV precisa ter cabeçalhos 'instruction' e 'output'. Encontrado: {fieldnames}"
                    )
                for line_no, row in enumerate(reader, start=2):
                    total_records += 1
                    is_valid, err = cls.validate_record(row, "alpaca", line_no)
                    if is_valid:
                        valid_records += 1
                        sample_texts.append(cls.extract_full_text(row, "alpaca"))
                    else:
                        invalid_records += 1
                        if len(errors) < max_errors and err:
                            errors.append(err)
        else:
            raise DatasetValidationError(f"Formato não suportado: {suffix}. Use .jsonl, .json ou .csv.")

        return {
            "file_name": path.name,
            "detected_schema": detected_schema,
            "total_records": total_records,
            "valid_records": valid_records,
            "invalid_records": invalid_records,
            "is_valid": invalid_records == 0 and total_records > 0,
            "errors_truncated": len(errors) == max_errors and invalid_records > max_errors,
            "errors": errors,
            "_internal_texts": sample_texts,  # stripped by anonymizer before outputting telemetry
        }
