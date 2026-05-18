"""
RAVANA v2 — Telegram Reporter
Delivers RAVANA status, alerts, and reports via Telegram.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ReportFormat:
    """Formatting options for Telegram messages."""
    max_length: int = 4000  # Telegram message limit
    use_emoji: bool = True
    use_markdown: bool = True


class TelegramReporter:
    """
    Formats and sends RAVANA cognitive state updates via Telegram.
    
    Formats:
    - Status cards (compact state summary)
    - Alert messages (urgent notifications)
    - Full reports (detailed analysis)
    - Insight cards (wisdom/learning highlights)
    """
    
    def __init__(self, send_fn=None, format_options: ReportFormat = None):
        """
        Args:
            send_fn: Function to send messages. If None, messages are printed.
            format_options: Formatting configuration.
        """
        self._send = send_fn
        self.fmt = format_options or ReportFormat()
    
    def send_status_card(self, state: dict, episode: int = None) -> str:
        """
        Send a compact status card showing current RAVANA state.
        
        Args:
            state: RAVANA state dict from get_state_vector()
            episode: Current episode number
        
        Returns:
            The formatted message string
        """
        d = state.get('dissonance', 0)
        i = state.get('identity', 0)
        w = state.get('wisdom', 0)
        mode = state.get('governor_mode', 'unknown')
        
        # Emoji map
        mode_emoji = {
            "normal": "⚖️",
            "exploration": "🔍",
            "resolution": "🔥",
            "recovery": "💚",
            "plateau": "⏸️",
            "unknown": "❓",
        }
        emoji = mode_emoji.get(mode, "⚙️")
        
        # Dissonance bar (10 segments)
        d_bar = "█" * int(d * 10) + "░" * (10 - int(d * 10))
        i_bar = "█" * int(i * 10) + "░" * (10 - int(i * 10))
        
        lines = []
        if self.fmt.use_emoji:
            lines.append(f"{emoji} RAVANA v2 Status")
        else:
            lines.append("RAVANA v2 Status")
        
        lines.append("─" * 20)
        
        if episode:
            lines.append(f"Episode: {episode}")
        
        lines.append(f"Dissonance: [{d_bar}] {d:.1%}")
        lines.append(f"Identity:   [{i_bar}] {i:.1%}")
        lines.append(f"Wisdom:     {w:.2f}")
        lines.append(f"Mode:       {mode.upper()}")
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def send_alert(self, message: str, severity: str = "info") -> str:
        """
        Send an urgent alert message.
        
        Args:
            message: Alert content
            severity: "info" | "warning" | "critical"
        
        Returns:
            Formatted message
        """
        emoji = {
            "info": "ℹ️",
            "warning": "⚠️",
            "critical": "🚨",
        }.get(severity, "ℹ️")
        
        lines = [
            f"{emoji} RAVANA Alert",
            "─" * 20,
            message,
        ]
        
        if severity == "critical":
            lines.append("")
            lines.append("⚠️ Immediate attention required")
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def send_report(self, report_type: str, data: dict) -> str:
        """
        Send a formatted report.
        
        Args:
            report_type: "grounding" | "insight" | "learning" | "diagnostic"
            data: Report data dict
        
        Returns:
            Formatted message
        """
        if report_type == "grounding":
            return self._format_grounding_report(data)
        elif report_type == "insight":
            return self._format_insight_report(data)
        elif report_type == "learning":
            return self._format_learning_report(data)
        elif report_type == "diagnostic":
            return self._format_diagnostic_report(data)
        else:
            return self._format_generic_report(data)
    
    def send_diagnostic_report(self, ravana_wrapper) -> str:
        """
        Generate and send a full system diagnostic.
        
        Args:
            ravana_wrapper: RavanaWrapper instance
        
        Returns:
            Formatted message
        """
        diagnosis = ravana_wrapper.get_diagnosis()
        state = ravana_wrapper.get_state_vector()
        clamp_metrics = ravana_wrapper.governor.get_clamp_metrics()
        
        lines = [
            "🧠 RAVANA v2 Diagnostic Report",
            "─" * 30,
            diagnosis,
            "",
            "📊 Governor Health:",
            f"  Clamp Rate: {clamp_metrics.get('clamp_rate', 0):.1%}",
            f"  Alignment:  {clamp_metrics.get('alignment_score', 0):.1%}",
            f"  Total Clamps: {clamp_metrics.get('total_clamps', 0)}",
        ]
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def send_episode_summary(self, episode_data: dict) -> str:
        """
        Send a summary of a completed episode.
        
        Args:
            episode_data: Dict from RavanaWrapper.step()
        
        Returns:
            Formatted message
        """
        ep = episode_data.get('episode', '?')
        mode = episode_data.get('mode', 'unknown')
        pre_d = episode_data.get('pre_dissonance', 0)
        post_d = episode_data.get('post_dissonance', 0)
        pre_i = episode_data.get('pre_identity', 0)
        post_i = episode_data.get('post_identity', 0)
        wisdom = episode_data.get('wisdom', 0)
        correct = episode_data.get('resolution', {}).get('full_resolution', False)
        
        # Direction arrows
        d_arrow = "↓" if post_d < pre_d else "↑"
        i_arrow = "↑" if post_i > pre_i else "↓"
        
        lines = [
            f"📋 Episode {ep} Complete",
            "─" * 25,
            f"Outcome: {'✅ Correct' if correct else '❌ Incorrect'}",
            f"Dissonance: {pre_d:.2f} → {post_d:.2f} {d_arrow}",
            f"Identity:   {pre_i:.2f} → {post_i:.2f} {i_arrow}",
            f"Wisdom: +{wisdom:.3f}" if wisdom > 0 else f"Wisdom: {wisdom:.3f}",
            f"Mode: {mode.upper()}",
        ]
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def send_wisdom_gain(self, wisdom_amount: float, reason: str, state: dict) -> str:
        """
        Send a wisdom generation notification.
        
        Args:
            wisdom_amount: How much wisdom was generated
            reason: Why wisdom was generated
            state: Current RAVANA state
        
        Returns:
            Formatted message
        """
        lines = [
            "✨ Wisdom Generated",
            "─" * 25,
            f"Amount: +{wisdom_amount:.3f}",
            f"Reason: {reason}",
            f"Current wisdom: {state.get('wisdom', 0):.2f}",
        ]
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    # ─── Private Formatting Methods ────────────────────────────────────────────
    
    def _format_grounding_report(self, data: dict) -> str:
        news_items = data.get('news', [])[:5]
        
        lines = [
            "🌍 Reality Grounding Report",
            "─" * 30,
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}",
            f"Articles analyzed: {len(news_items)}",
            "",
            "TOP STORIES:",
        ]
        
        for i, item in enumerate(news_items, 1):
            lines.append(f"{i}. {item.get('title', 'No title')[:80]}")
            lines.append(f"   {item.get('summary', '')[:100]}...")
        
        alignment = data.get('alignment_check', {})
        if alignment:
            lines.append("")
            lines.append(f"Alignment: {alignment.get('verdict', 'unknown').upper()}")
            lines.append(f"Score: {alignment.get('alignment_score', 0):.2f}")
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def _format_insight_report(self, data: dict) -> str:
        insight = data.get('insight', 'No insight available')
        state = data.get('state', {})
        
        lines = [
            "💡 RAVANA Insight",
            "─" * 30,
            insight,
            "",
            f"Episode: {state.get('episode', '?')} | D: {state.get('dissonance', 0):.2f} | I: {state.get('identity', 0):.2f}",
        ]
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def _format_learning_report(self, data: dict) -> str:
        lessons = data.get('lessons', [])
        
        lines = [
            "📚 Learning Report",
            "─" * 30,
            f"Lessons learned: {len(lessons)}",
            "",
        ]
        
        for lesson in lessons[-5:]:
            lines.append(f"• {lesson.get('lesson', 'No description')[:100]}")
            lines.append(f"  Confidence: {lesson.get('confidence', 0):.0%}")
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def _format_diagnostic_report(self, data: dict) -> str:
        lines = [
            "🔧 Diagnostic Report",
            "─" * 30,
        ]
        
        for key, value in data.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.3f}")
            else:
                lines.append(f"{key}: {value}")
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg
    
    def _format_generic_report(self, data: dict) -> str:
        lines = [
            "📊 RAVANA Report",
            "─" * 30,
        ]
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}: ({len(value)} items)")
            else:
                lines.append(f"{key}: {value}")
        
        msg = "\n".join(lines)
        
        if self._send:
            self._send(msg)
        
        return msg


if __name__ == "__main__":
    # Test formatting (no actual sending)
    reporter = TelegramReporter(send_fn=None)
    
    print("=== Telegram Reporter Test ===\n")
    
    # Test status card
    test_state = {
        'dissonance': 0.55,
        'identity': 0.68,
        'wisdom': 15.3,
        'governor_mode': 'resolution',
        'episode': 42,
    }
    print(reporter.send_status_card(test_state, episode=42))
    print()
    
    # Test episode summary
    episode = {
        'episode': 42,
        'mode': 'resolution',
        'pre_dissonance': 0.60,
        'post_dissonance': 0.52,
        'pre_identity': 0.65,
        'post_identity': 0.68,
        'wisdom': 0.15,
        'resolution': {'full_resolution': True},
    }
    print(reporter.send_episode_summary(episode))
    print()
    
    # Test alert
    print(reporter.send_alert("Dissonance spike detected: D=0.89", severity="warning"))