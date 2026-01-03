"""Tests for CLI server command."""

from unittest.mock import patch

from click.testing import CliRunner

from lumina.cli.server import server


class TestServerCommand:
    """Tests for server CLI command."""

    def test_server_command_default_options(self):
        """Test server command with default options."""
        runner = CliRunner()

        with patch("lumina.cli.server.uvicorn.run") as mock_run:
            result = runner.invoke(server, [], catch_exceptions=False)

            # Command should execute successfully
            assert result.exit_code == 0

            # Should call uvicorn.run with default options
            mock_run.assert_called_once_with(
                "lumina.api.app:create_app",
                factory=True,
                host="0.0.0.0",
                port=8000,
                reload=False,
                log_level="info",
            )

    def test_server_command_custom_host_port(self):
        """Test server command with custom host and port."""
        runner = CliRunner()

        with patch("lumina.cli.server.uvicorn.run") as mock_run:
            result = runner.invoke(
                server,
                ["--host", "127.0.0.1", "--port", "9000"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_run.assert_called_once_with(
                "lumina.api.app:create_app",
                factory=True,
                host="127.0.0.1",
                port=9000,
                reload=False,
                log_level="info",
            )

    def test_server_command_with_reload(self):
        """Test server command with reload flag."""
        runner = CliRunner()

        with patch("lumina.cli.server.uvicorn.run") as mock_run:
            result = runner.invoke(server, ["--reload"], catch_exceptions=False)

            assert result.exit_code == 0
            mock_run.assert_called_once_with(
                "lumina.api.app:create_app",
                factory=True,
                host="0.0.0.0",
                port=8000,
                reload=True,
                log_level="info",
            )

    def test_server_command_all_options(self):
        """Test server command with all options."""
        runner = CliRunner()

        with patch("lumina.cli.server.uvicorn.run") as mock_run:
            result = runner.invoke(
                server,
                ["--host", "localhost", "--port", "5000", "--reload"],
                catch_exceptions=False,
            )

            assert result.exit_code == 0
            mock_run.assert_called_once_with(
                "lumina.api.app:create_app",
                factory=True,
                host="localhost",
                port=5000,
                reload=True,
                log_level="info",
            )
