"""Import all ORM models so Alembic can auto-discover them."""
from app.db.models.conversation import Conversation  # noqa: F401
from app.db.models.feedback import Feedback  # noqa: F401
from app.db.models.interaction_log import InteractionLog  # noqa: F401
from app.db.models.message import Message  # noqa: F401
from app.db.models.app_setting import AppSetting  # noqa: F401
from app.db.models.message_translation import MessageTranslation  # noqa: F401
