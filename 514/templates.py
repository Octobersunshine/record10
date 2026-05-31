import os
import logging
from jinja2 import Environment, FileSystemLoader, select_autoescape, Template

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemplateRenderer:
    def __init__(self, template_dir='templates'):
        self.template_dir = template_dir
        self._env = None
        self._init_environment()

    def _init_environment(self):
        if os.path.exists(self.template_dir):
            self._env = Environment(
                loader=FileSystemLoader(self.template_dir),
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )
            logger.info(f"Template environment initialized with directory: {self.template_dir}")
        else:
            logger.warning(f"Template directory {self.template_dir} not found, using inline templates only")
            self._env = Environment(
                autoescape=select_autoescape(['html', 'xml']),
                trim_blocks=True,
                lstrip_blocks=True
            )

    def render_template(self, template_name, context=None):
        if context is None:
            context = {}

        if self._env and self._env.loader:
            try:
                template = self._env.get_template(template_name)
                return template.render(**context)
            except Exception as e:
                logger.error(f"Failed to render template {template_name}: {e}")
                raise
        else:
            raise FileNotFoundError(f"Template directory not available, cannot load template: {template_name}")

    def render_string(self, template_string, context=None):
        if context is None:
            context = {}

        try:
            template = Template(template_string)
            return template.render(**context)
        except Exception as e:
            logger.error(f"Failed to render inline template: {e}")
            raise

    def list_templates(self):
        if self._env and self._env.loader:
            try:
                return self._env.list_templates()
            except Exception as e:
                logger.error(f"Failed to list templates: {e}")
                return []
        return []


template_renderer = TemplateRenderer()


def render_email_template(template_name=None, template_string=None, context=None):
    if template_name:
        return template_renderer.render_template(template_name, context)
    elif template_string:
        return template_renderer.render_string(template_string, context)
    else:
        raise ValueError("Either template_name or template_string must be provided")


def get_available_templates():
    return template_renderer.list_templates()
