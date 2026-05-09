import typer
from rich.console import Console
from rich.panel import Panel
from core.profile.commands import register_command
console = Console()
app = typer.Typer(help="🕵️ Holmz — your model interrogation companion.")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Holmz — The Sherlock Holmes of AI explainability."""
    if ctx.invoked_subcommand is None:
        console.print(
            Panel.fit(
                "[bold cyan]Holmz 🧩[/bold cyan]\n"
                "[yellow]Every prediction has a story — uncover yours.[/yellow]\n\n"
                "Holmz helps you interrogate Machine Learning and Deep Learning models,\n"
                "revealing clear and transparent explanations for their behavior.\n",
                title="🕵️ Welcome to Holmz CLI",
                border_style="bright_magenta",
            )
        )
        console.print("Use [bold green]holmz --help[/bold green] to see available commands.\n")


app.command(name='register')(register_command)
if __name__ == "__main__":
    app()
