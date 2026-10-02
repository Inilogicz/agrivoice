"""Repository for runtime-editable app settings."""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models.app_setting import AppSetting


class AppSettingsRepository:
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def get(self, key: str) -> AppSetting | None:
        return await self.session.get(AppSetting, key)

    async def set(self, key: str, value: str) -> AppSetting:
        setting = await self.session.get(AppSetting, key)
        if setting is None:
            setting = AppSetting(key=key, value=value)
            self.session.add(setting)
        else:
            setting.value = value
        await self.session.flush()
        await self.session.refresh(setting)
        return setting

    async def delete(self, key: str) -> None:
        setting = await self.session.get(AppSetting, key)
        if setting is not None:
            await self.session.delete(setting)
            await self.session.flush()
