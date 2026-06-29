from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from typing import Any

import pandas as pd

from core.observation.collectors.missingness import (
	ColumnMissingnessObservationCollector,
	ColumnsMissingnessDistributionObservationCollector,
	RowsMissingnessDistributionObservationCollector,
	RowsMissingnessObservationCollector,
)
from core.profilers.missingness.pandas_profiler import PandasMissingRateProfiler


@dataclass
class _InspectContext:
	dataset: pd.DataFrame
	sample_size: int


@dataclass
class InspectionReport:
	overall_health: float
	critical_columns: list[dict[str, Any]]
	moderate_columns: list[dict[str, Any]]
	healthy_columns: list[str]
	full_missing_rows_rate: float
	high_missing_rows_rate: float

	def _format_percent(self, value: float) -> str:
		return f"{value * 100:.0f}%"

	def _build_text(self) -> str:
		lines: list[str] = [
			"Dataset Health Report",
			"---------------------",
			"",
			f"Overall Health: {self.overall_health:.2f}",
			"",
		]

		if self.critical_columns:
			title = "Critical Issue:" if len(self.critical_columns) == 1 else "Critical Issues:"
			lines.append(title)
			for item in self.critical_columns:
				lines.append(
					f"- {item['column_name']} -> {self._format_percent(item['missing_rate'])} missing"
				)
		else:
			lines.extend(["Critical Issue:", "- none"])

		lines.append("")
		if self.moderate_columns:
			lines.append("Moderate Issues:")
			for item in self.moderate_columns:
				lines.append(
					f"- {item['column_name']} -> {self._format_percent(item['missing_rate'])} missing"
				)
		else:
			lines.extend(["Moderate Issues:", "- none"])

		lines.append("")
		lines.append("Healthy:")
		if self.healthy_columns:
			for column in self.healthy_columns:
				lines.append(f"- {column}")
		else:
			lines.append("- none")

		lines.append("")
		lines.append("Evidence:")
		lines.append(f"- fully missing rows: {self._format_percent(self.full_missing_rows_rate)}")
		lines.append(
			f"- highly missing rows (>=80% null): {self._format_percent(self.high_missing_rows_rate)}"
		)

		return "\n".join(lines)

	def show(self) -> None:
		print(self._build_text())

	def _repr_markdown_(self) -> str:
		return self._build_text()

	def _health_band(self) -> str:
		if self.overall_health >= 0.85:
			return "Healthy"
		if self.overall_health >= 0.65:
			return "Watch"
		return "Fragile"

	@staticmethod
	def _escape_html(value: str) -> str:
		return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

	def _render_issue_rows(self, rows: list[dict[str, Any]], tone: str) -> str:
		if not rows:
			return "<li class='hm-empty'>none</li>"

		items: list[str] = []
		for item in rows:
			name = self._escape_html(str(item["column_name"]))
			rate = self._format_percent(float(item["missing_rate"]))
			items.append(
				f"<li><span class='hm-col'>{name}</span><span class='hm-rate hm-rate-{tone}'>{rate}</span></li>"
			)
		return "".join(items)

	def _render_healthy_rows(self) -> str:
		if not self.healthy_columns:
			return "<li class='hm-empty'>none</li>"
		return "".join(f"<li>{self._escape_html(col)}</li>" for col in self.healthy_columns)

	def _repr_html_(self) -> str:
		health_pct = self._format_percent(self.overall_health)
		band = self._health_band()
		critical_rows = self._render_issue_rows(self.critical_columns, "critical")
		moderate_rows = self._render_issue_rows(self.moderate_columns, "moderate")
		healthy_rows = self._render_healthy_rows()
		full_rows = self._format_percent(self.full_missing_rows_rate)
		high_rows = self._format_percent(self.high_missing_rows_rate)

		return f"""
<div class="hm-shell">
	<style>
		.hm-shell {{
			--hm-ink: #1f1f24;
			--hm-muted: #6f727a;
			--hm-bg: #fbfaf7;
			--hm-panel: #ffffff;
			--hm-line: #e7e3da;
			--hm-critical: #9f2b2b;
			--hm-moderate: #8b5a1d;
			--hm-ok: #2f6f4f;
			font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
			color: var(--hm-ink);
			max-width: 820px;
			border: 1px solid var(--hm-line);
			background:
				radial-gradient(circle at 12px 12px, rgba(31,31,36,0.03) 1px, transparent 1px) 0 0/18px 18px,
				linear-gradient(145deg, #fcfbf8 0%, #f7f3ea 100%);
			border-radius: 14px;
			overflow: hidden;
			box-shadow: 0 8px 24px rgba(20, 20, 30, 0.06);
		}}
		.hm-head {{
			display: flex;
			align-items: end;
			justify-content: space-between;
			padding: 18px 20px 14px;
			border-bottom: 1px solid var(--hm-line);
			background: var(--hm-panel);
		}}
		.hm-title {{
			font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
			font-size: 13px;
			letter-spacing: 0.08em;
			text-transform: uppercase;
			color: var(--hm-muted);
			margin: 0;
		}}
		.hm-score {{
			margin: 4px 0 0;
			font-size: 30px;
			font-weight: 700;
			line-height: 1;
		}}
		.hm-band {{
			font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
			font-size: 12px;
			letter-spacing: 0.06em;
			text-transform: uppercase;
			color: var(--hm-muted);
			border: 1px solid var(--hm-line);
			border-radius: 999px;
			padding: 7px 12px;
			background: #fff;
		}}
		.hm-grid {{
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
			gap: 10px;
			padding: 12px;
		}}
		.hm-card {{
			border: 1px solid var(--hm-line);
			border-radius: 10px;
			padding: 10px 12px 11px;
			background: var(--hm-panel);
		}}
		.hm-card h4 {{
			margin: 0 0 8px;
			font-size: 12px;
			letter-spacing: 0.07em;
			text-transform: uppercase;
			color: var(--hm-muted);
			font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
		}}
		.hm-card ul {{
			margin: 0;
			padding-left: 16px;
		}}
		.hm-card li {{
			display: flex;
			justify-content: space-between;
			gap: 10px;
			font-size: 13px;
			line-height: 1.55;
			margin: 2px 0;
		}}
		.hm-col {{
			color: var(--hm-ink);
			font-weight: 500;
			word-break: break-word;
		}}
		.hm-rate {{
			font-family: "IBM Plex Mono", "JetBrains Mono", monospace;
			font-size: 12px;
			border-radius: 999px;
			padding: 1px 8px;
			white-space: nowrap;
			border: 1px solid transparent;
		}}
		.hm-rate-critical {{
			color: var(--hm-critical);
			background: rgba(159, 43, 43, 0.08);
			border-color: rgba(159, 43, 43, 0.2);
		}}
		.hm-rate-moderate {{
			color: var(--hm-moderate);
			background: rgba(139, 90, 29, 0.1);
			border-color: rgba(139, 90, 29, 0.24);
		}}
		.hm-empty {{
			color: var(--hm-muted);
			font-style: italic;
		}}
		.hm-evidence {{
			border-top: 1px solid var(--hm-line);
			padding: 12px 16px 14px;
			background: #fff;
			font-size: 13px;
			color: var(--hm-muted);
		}}
		.hm-evidence strong {{
			color: var(--hm-ink);
			font-weight: 600;
		}}
		@media (max-width: 560px) {{
			.hm-head {{
				flex-direction: column;
				align-items: start;
				gap: 10px;
			}}
			.hm-score {{
				font-size: 26px;
			}}
		}}
	</style>

	<header class="hm-head">
		<div>
			<p class="hm-title">Dataset Health Report</p>
			<p class="hm-score">{health_pct} health</p>
		</div>
		<span class="hm-band">{band}</span>
	</header>

	<section class="hm-grid">
		<article class="hm-card">
			<h4>Critical</h4>
			<ul>{critical_rows}</ul>
		</article>
		<article class="hm-card">
			<h4>Moderate</h4>
			<ul>{moderate_rows}</ul>
		</article>
		<article class="hm-card">
			<h4>Healthy</h4>
			<ul>{healthy_rows}</ul>
		</article>
	</section>

	<footer class="hm-evidence">
		<strong>Evidence:</strong>
		fully missing rows {full_rows} | highly missing rows (&gt;=80% null) {high_rows}
	</footer>
</div>
"""

	def __str__(self) -> str:
		return self._build_text()


def inspect(
	df: pd.DataFrame,
	*,
	sample_size: int = 25,
	critical_missing_threshold: float = 0.60,
	moderate_missing_threshold: float = 0.20,
	healthy_preview_limit: int = 12,
) -> InspectionReport:
	"""Inspect a dataframe for missing-value quality issues.

	The API is intentionally simple and notebook-friendly: one call returns a
	human-readable report with a compact summary of what broke and where.
	"""
	if not isinstance(df, pd.DataFrame):
		raise TypeError("inspect(df) expects a pandas.DataFrame")

	if len(df.columns) == 0:
		return InspectionReport(
			overall_health=1.0,
			critical_columns=[],
			moderate_columns=[],
			healthy_columns=[],
			full_missing_rows_rate=0.0,
			high_missing_rows_rate=0.0,
		)

	profiler = PandasMissingRateProfiler()
	context = _InspectContext(dataset=df, sample_size=max(1, int(sample_size)))

	column_observations = ColumnMissingnessObservationCollector(profiler).collect(context)
	rows_observation = RowsMissingnessObservationCollector(profiler).collect(context)[0]
	_ = RowsMissingnessDistributionObservationCollector(profiler).collect(context)[0]
	columns_distribution_observation = (
		ColumnsMissingnessDistributionObservationCollector(profiler).collect(context)[0]
	)

	column_profiles = [obs.payload for obs in column_observations]
	column_profiles.sort(key=lambda item: item["missing_rate"], reverse=True)

	critical_columns = [
		item for item in column_profiles if item["missing_rate"] >= critical_missing_threshold
	]
	moderate_columns = [
		item
		for item in column_profiles
		if moderate_missing_threshold <= item["missing_rate"] < critical_missing_threshold
	]
	healthy_columns = [
		item["column_name"] for item in column_profiles if item["missing_rate"] < moderate_missing_threshold
	][:healthy_preview_limit]

	rows_payload = rows_observation.payload
	cols_dist_payload = columns_distribution_observation.payload

	mean_missing = float(cols_dist_payload.get("mean_missing_rate", 0.0) or 0.0)
	worst_missing = float(column_profiles[0]["missing_rate"]) if column_profiles else 0.0
	full_missing_rows_rate = float(rows_payload.get("full_missing_rows_rate", 0.0) or 0.0)
	high_missing_rows_rate = float(rows_payload.get("high_missing_rate_rows_rate", 0.0) or 0.0)

	if isnan(mean_missing):
		mean_missing = 0.0

	# Weighted blend optimized for fast interpretability in notebooks.
	degradation = (
		(0.50 * mean_missing)
		+ (0.30 * worst_missing)
		+ (0.10 * full_missing_rows_rate)
		+ (0.10 * high_missing_rows_rate)
	)
	overall_health = max(0.0, min(1.0, 1.0 - degradation))

	return InspectionReport(
		overall_health=overall_health,
		critical_columns=critical_columns,
		moderate_columns=moderate_columns,
		healthy_columns=healthy_columns,
		full_missing_rows_rate=full_missing_rows_rate,
		high_missing_rows_rate=high_missing_rows_rate,
	)
