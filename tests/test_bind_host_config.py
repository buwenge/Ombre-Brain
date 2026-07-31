"""监听地址必须可由环境变量收紧，同时保持容器默认行为。"""

from pathlib import Path
import unittest


_ROOT = Path(__file__).resolve().parents[1]
_SERVER = _ROOT / "server.py"


class BindHostConfigTests(unittest.TestCase):
    def test_bind_host_uses_env_with_container_compatible_default(self):
        source = _SERVER.read_text(encoding="utf-8")

        self.assertIn('os.environ.get("OMBRE_BIND_HOST", "0.0.0.0")', source)
        self.assertEqual(source.count("host=OMBRE_BIND_HOST"), 2)

    def test_bind_host_is_documented(self):
        docs = (_ROOT / "ENV_VARS.md").read_text(encoding="utf-8")

        self.assertIn("`OMBRE_BIND_HOST`", docs)
        self.assertIn("`127.0.0.1`", docs)


if __name__ == "__main__":
    unittest.main()
