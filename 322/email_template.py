import os
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Any

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound, TemplateSyntaxError

logger = logging.getLogger(__name__)


@dataclass
class RenderedEmail:
    subject: str
    body: str
    is_html: bool = True


class EmailTemplate:
    def __init__(self, template_dir: Optional[str] = None):
        if template_dir:
            self._env = Environment(
                loader=FileSystemLoader(template_dir),
                autoescape=True,
            )
        else:
            self._env = Environment(autoescape=True)
        self._template_dir = template_dir

    def render_string(self, template_str: str, context: Dict[str, Any]) -> str:
        try:
            template = self._env.from_string(template_str)
            return template.render(**context)
        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error: {e}")
            raise ValueError(f"Template syntax error: {e}") from e

    def render_file(self, template_name: str, context: Dict[str, Any]) -> str:
        if not self._template_dir:
            raise ValueError("template_dir is required for file-based templates")
        try:
            template = self._env.get_template(template_name)
            return template.render(**context)
        except TemplateNotFound as e:
            logger.error(f"Template not found: {e}")
            raise FileNotFoundError(f"Template not found: {template_name}") from e
        except TemplateSyntaxError as e:
            logger.error(f"Template syntax error in {template_name}: {e}")
            raise ValueError(f"Template syntax error in {template_name}: {e}") from e

    def render_email(
        self,
        subject_template: str,
        body_template: str,
        context: Dict[str, Any],
        is_html: bool = True,
        template_from_file: bool = False,
    ) -> RenderedEmail:
        if template_from_file:
            subject = self.render_file(subject_template, context)
            body = self.render_file(body_template, context)
        else:
            subject = self.render_string(subject_template, context)
            body = self.render_string(body_template, context)

        return RenderedEmail(subject=subject, body=body, is_html=is_html)
