"""Simplified logger for unit tests."""


class _Logger:
    def info(self, message):
        print(f"[INFO] {message}")

    def warning(self, message):
        print(f"[WARNING] {message}")

    def critical(self, message):
        print(f"[CRITICAL] {message}")


logger = _Logger()

