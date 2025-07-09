#!/usr/bin/env python3
"""
Slack API 클라이언트 구현
FastMCP를 위한 Slack API 연동 클라이언트
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any, Union
import json
import logging

logger = logging.getLogger(__name__)


class SlackAPIError(Exception):
    """Slack API 에러 클래스"""
    
    def __init__(self, error: str, response: Optional[Dict] = None):
        self.error = error
        self.response = response
        super().__init__(f"Slack API Error: {error}")


class SlackAPIClient:
    """
    Slack API 클라이언트 클래스
    
    이 클래스는 Slack API와의 모든 상호작용을 담당합니다.
    비동기 HTTP 요청을 사용하여 성능을 최적화합니다.
    """
    
    def __init__(self, token: str):
        """
        Slack API 클라이언트 초기화
        
        Args:
            token (str): Slack Bot Token (xoxb-로 시작)
        """
        self.token = token
        self.base_url = "https://slack.com/api"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "FastMCP-Slack-Client/1.0"
        }
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """
        HTTP 세션 획득 (단일 세션 재사용)
        
        Returns:
            aiohttp.ClientSession: HTTP 세션
        """
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=30)
            )
        return self._session
    
    async def close(self) -> None:
        """
        HTTP 세션 정리
        """
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Slack API에 HTTP 요청을 보내는 내부 메서드
        
        Args:
            method (str): HTTP 메서드 (GET, POST 등)
            endpoint (str): API 엔드포인트
            data (Dict, optional): 요청 본문 데이터
            params (Dict, optional): URL 파라미터
            
        Returns:
            Dict[str, Any]: API 응답 데이터
            
        Raises:
            SlackAPIError: API 요청 실패 시
        """
        session = await self._get_session()
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if method.upper() == "GET":
                async with session.get(url, params=params) as response:
                    result = await response.json()
            else:
                async with session.post(url, json=data, params=params) as response:
                    result = await response.json()
            
            if not result.get("ok", False):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"Slack API error: {error_msg}")
                raise SlackAPIError(error_msg, result)
            
            return result
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e}")
            raise SlackAPIError(f"HTTP request failed: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode JSON response: {e}")
            raise SlackAPIError(f"Invalid JSON response: {e}")
    
    async def send_message(self, channel: str, text: str) -> Dict[str, Any]:
        """
        Slack 채널에 메시지 전송
        
        Args:
            channel (str): 채널 ID 또는 채널명 (예: #general, C1234567890)
            text (str): 전송할 메시지 내용
            
        Returns:
            Dict[str, Any]: 메시지 전송 결과
            
        Raises:
            SlackAPIError: 메시지 전송 실패 시
        """
        data = {
            "channel": channel,
            "text": text,
            "link_names": True,
            "parse": "full"
        }
        
        logger.info(f"Sending message to channel {channel}")
        return await self._make_request("POST", "chat.postMessage", data)
    
    async def get_channels(self, exclude_archived: bool = True) -> List[Dict[str, Any]]:
        """
        접근 가능한 모든 채널 목록 조회
        
        Args:
            exclude_archived (bool): 아카이브된 채널 제외 여부
            
        Returns:
            List[Dict[str, Any]]: 채널 목록
            
        Raises:
            SlackAPIError: 채널 목록 조회 실패 시
        """
        params = {
            "exclude_archived": exclude_archived,
            "types": "public_channel,private_channel"
        }
        
        logger.info("Fetching channel list")
        result = await self._make_request("GET", "conversations.list", params=params)
        return result.get("channels", [])
    
    async def get_channel_history(
        self, 
        channel_id: str, 
        limit: int = 10, 
        oldest: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        지정된 채널의 최근 메시지 히스토리 조회
        
        Args:
            channel_id (str): 조회할 채널의 ID
            limit (int): 조회할 메시지 수 (기본값: 10, 최대: 100)
            oldest (str, optional): 이 시점 이후의 메시지만 조회
            
        Returns:
            List[Dict[str, Any]]: 메시지 목록
            
        Raises:
            SlackAPIError: 히스토리 조회 실패 시
        """
        params = {
            "channel": channel_id,
            "limit": min(limit, 100)  # 최대 100개로 제한
        }
        
        if oldest:
            params["oldest"] = oldest
        
        logger.info(f"Fetching history for channel {channel_id}")
        result = await self._make_request("GET", "conversations.history", params=params)
        return result.get("messages", [])
    
    async def open_dm_channel(self, user_id: str) -> str:
        """
        사용자와의 DM 채널 열기
        
        Args:
            user_id (str): 사용자 ID
            
        Returns:
            str: DM 채널 ID
            
        Raises:
            SlackAPIError: DM 채널 열기 실패 시
        """
        data = {
            "users": user_id
        }
        
        logger.info(f"Opening DM channel with user {user_id}")
        result = await self._make_request("POST", "conversations.open", data)
        channel = result.get("channel", {})
        return channel.get("id", "")
    
    async def send_direct_message(self, user_id: str, text: str) -> Dict[str, Any]:
        """
        특정 사용자에게 1:1 다이렉트 메시지 전송
        
        Args:
            user_id (str): 메시지를 받을 사용자의 ID
            text (str): 전송할 메시지 내용
            
        Returns:
            Dict[str, Any]: 메시지 전송 결과
            
        Raises:
            SlackAPIError: 다이렉트 메시지 전송 실패 시
        """
        # 먼저 DM 채널을 열어야 함
        dm_channel_id = await self.open_dm_channel(user_id)
        
        if not dm_channel_id:
            raise SlackAPIError("Failed to open DM channel")
        
        # DM 채널에 메시지 전송
        return await self.send_message(dm_channel_id, text)
    
    async def get_users(self) -> List[Dict[str, Any]]:
        """
        워크스페이스의 모든 사용자 정보 조회
        
        Returns:
            List[Dict[str, Any]]: 사용자 목록
            
        Raises:
            SlackAPIError: 사용자 목록 조회 실패 시
        """
        logger.info("Fetching user list")
        result = await self._make_request("GET", "users.list")
        return result.get("members", [])
    
    async def add_reaction(self, channel: str, timestamp: str, name: str) -> Dict[str, Any]:
        """
        특정 메시지에 이모지 반응 추가
        
        Args:
            channel (str): 채널 ID
            timestamp (str): 메시지 타임스탬프
            name (str): 이모지 이름 (콜론 없이, 예: 'thumbsup')
            
        Returns:
            Dict[str, Any]: 반응 추가 결과
            
        Raises:
            SlackAPIError: 반응 추가 실패 시
        """
        data = {
            "channel": channel,
            "timestamp": timestamp,
            "name": name
        }
        
        logger.info(f"Adding reaction {name} to message {timestamp}")
        return await self._make_request("POST", "reactions.add", data)
    
    async def search_messages(self, query: str, count: int = 20) -> List[Dict[str, Any]]:
        """
        키워드를 통한 메시지 검색
        
        Args:
            query (str): 검색 키워드
            count (int): 반환할 결과 수 (최대 100)
            
        Returns:
            List[Dict[str, Any]]: 검색 결과
            
        Raises:
            SlackAPIError: 검색 실패 시
        """
        params = {
            "query": query,
            "count": min(count, 100),
            "sort": "timestamp"
        }
        
        logger.info(f"Searching messages with query: {query}")
        result = await self._make_request("GET", "search.messages", params=params)
        
        # 검색 결과 구조 처리
        messages = result.get("messages", {})
        return messages.get("matches", [])
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Slack API 연결 테스트
        
        Returns:
            Dict[str, Any]: 연결 테스트 결과
            
        Raises:
            SlackAPIError: 연결 테스트 실패 시
        """
        logger.info("Testing Slack API connection")
        return await self._make_request("POST", "auth.test")


# 모듈 레벨에서 클라이언트 인스턴스를 관리하기 위한 전역 변수
_slack_client: Optional[SlackAPIClient] = None


def get_slack_client(token: str) -> SlackAPIClient:
    """
    Slack 클라이언트 싱글톤 인스턴스 반환
    
    Args:
        token (str): Slack Bot Token
        
    Returns:
        SlackAPIClient: Slack API 클라이언트 인스턴스
    """
    global _slack_client
    
    if _slack_client is None:
        _slack_client = SlackAPIClient(token)
    
    return _slack_client


async def cleanup_slack_client() -> None:
    """
    Slack 클라이언트 정리
    """
    global _slack_client
    
    if _slack_client:
        await _slack_client.close()
        _slack_client = None 