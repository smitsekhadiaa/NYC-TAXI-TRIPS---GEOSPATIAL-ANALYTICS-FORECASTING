"""Flask CLI commands for backend service."""

import click
from flask import Flask


def register_cli_commands(app: Flask) -> None:
    
    @app.cli.command("hello")
    def hello_command():
        """Simple test command."""
        click.echo("CLI is working!")