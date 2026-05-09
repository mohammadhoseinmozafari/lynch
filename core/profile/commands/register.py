from pathlib import Path
from typing import Optional
import typer
from core.case.register import RegisterModelRequest

def register_command (
        file : Optional[Path] = typer.Option(
            None, "--file", "-f",
            help="Path to JSON file containing registration data"

        ),
        data : Optional[str] = typer.Option(
            None, "--data", '-d',
            help="Registration data as JSON string"
        )
) :
    """
    Register a model with Holmz
    Provide registration data as a JSON file or JSON string.
    The JSON must match the RegisterModelRequest schema.
    Examples:
        holmz register --file registration.json
        holmz register --data '{"project_id" : "...", ...}'
    """
    json_str = _read_input(file, data)
    try:
        request= RegisterModelRequest.model_validate_json(json_str)
        
    except Exception as e :
        typer.echo(f"Invalid registration data:\n{e}", err=True)
        raise typer.Exit(code=1)
    

def _read_input(file: Optional[Path], data : Optional[str]) -> str :
    
    if file:
        return file.read_text()
    elif data :
        return data
    raise typer.BadParameter(
        "No input provided"
    )