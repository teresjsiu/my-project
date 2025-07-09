#!/usr/bin/env python3
"""
FastMCP를 이용한 Slack MCP 서버 구현
"""

import os
import sys
import asyncio
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path

# 환경 변수 로드
from dotenv import load_dotenv
load_dotenv()

# FastMCP 및 사용자 정의 모듈 임포트
try:
    from fastmcp import FastMCP
except ImportError:
    print("❌ FastMCP가 설치되지 않았습니다. pip install fastmcp를 실행하세요.")
    sys.exit(1)

from slack_api import get_slack_client, cleanup_slack_client, SlackAPIError

# # 로깅 설정
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.StreamHandler(),
#         logging.FileHandler('slack_mcp.log')
#     ]
# )
# logger = logging.getLogger(__name__)

# FastMCP 인스턴스 생성
mcp = FastMCP("Slack MCP Server")

# Slack Bot Token 확인
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
if not SLACK_BOT_TOKEN:
    print("SLACK_BOT_TOKEN 환경 변수가 설정되지 않았습니다.")
    sys.exit(1)

if not SLACK_BOT_TOKEN.startswith("xoxb-"):
    print("올바른 Slack Bot Token이 아닙니다. xoxb-로 시작해야 합니다.")
    sys.exit(1)

# Slack 클라이언트 초기화
slack_client = get_slack_client(SLACK_BOT_TOKEN)

@mcp.tool()
async def send_slack_message(channel: str, text: str) -> Dict[str, Any]:
    """
    Slack 채널에 메시지 전송
    
    Args:
        channel (str): 채널 ID 또는 채널명 (예: #general, C1234567890)
        text (str): 전송할 메시지 내용
        
    Returns:
        Dict[str, Any]: 메시지 전송 결과
    """
    try:
        # logger.info(f"Sending message to channel: {channel}")
        result = await slack_client.send_message(channel, text)
        
        # 성공 응답 포맷팅
        return {
            "success": True,
            "message": "메시지가 성공적으로 전송되었습니다.",
            "data": {
                "channel": result.get("channel"),
                "timestamp": result.get("ts"),
                "message_text": text
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in send_message: {e}")
        return {
            "success": False,
            "error": f"메시지 전송 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        # logger.error(f"Unexpected error in send_message: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

@mcp.tool()
async def get_slack_channels() -> Dict[str, Any]:
    """
    접근 가능한 모든 채널 목록 조회
    
    Returns:
        Dict[str, Any]: 채널 목록 및 정보
    """
    try:
        # logger.info("Fetching Slack channels")
        channels = await slack_client.get_channels()
        
        # 채널 정보 포맷팅
        formatted_channels = []
        for channel in channels:
            formatted_channels.append({
                "id": channel.get("id"),
                "name": channel.get("name"),
                "is_private": channel.get("is_private", False),
                "is_member": channel.get("is_member", False),
                "is_archived": channel.get("is_archived", False),
                "num_members": channel.get("num_members", 0),
                "topic": channel.get("topic", {}).get("value", ""),
                "purpose": channel.get("purpose", {}).get("value", "")
            })
        
        return {
            "success": True,
            "message": f"총 {len(formatted_channels)}개의 채널을 찾았습니다.",
            "data": {
                "channel_count": len(formatted_channels),
                "channels": formatted_channels
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in get_channels: {e}")
        return {
            "success": False,
            "error": f"채널 목록 조회 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        # logger.error(f"Unexpected error in get_channels: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

@mcp.tool()
async def get_slack_channel_history(channel_id: str, limit: int = 10) -> Dict[str, Any]:
    """
    지정된 채널의 최근 메시지 히스토리 조회
    
    Args:
        channel_id (str): 조회할 채널의 ID
        limit (int): 조회할 메시지 수 (기본값: 10, 최대: 100)
        
    Returns:
        Dict[str, Any]: 메시지 히스토리
    """
    try:
        # limit 값 검증 및 조정
        limit = max(1, min(limit, 100))
        
        # logger.info(f"Fetching history for channel: {channel_id}, limit: {limit}")
        messages = await slack_client.get_channel_history(channel_id, limit)
        
        # 메시지 정보 포맷팅
        formatted_messages = []
        for message in messages:
            formatted_messages.append({
                "timestamp": message.get("ts"),
                "user": message.get("user"),
                "text": message.get("text", ""),
                "type": message.get("type"),
                "subtype": message.get("subtype"),
                "edited": message.get("edited"),
                "reactions": message.get("reactions", [])
            })
        
        return {
            "success": True,
            "message": f"채널 {channel_id}에서 {len(formatted_messages)}개의 메시지를 조회했습니다.",
            "data": {
                "channel_id": channel_id,
                "message_count": len(formatted_messages),
                "messages": formatted_messages
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in get_channel_history: {e}")
        return {
            "success": False,
            "error": f"채널 히스토리 조회 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        # logger.error(f"Unexpected error in get_channel_history: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

@mcp.tool()
async def send_slack_direct_message(user_id: str, text: str) -> Dict[str, Any]:
    """
    특정 사용자에게 1:1 다이렉트 메시지 전송
    
    Args:
        user_id (str): 메시지를 받을 사용자의 ID
        text (str): 전송할 메시지 내용
        
    Returns:
        Dict[str, Any]: 다이렉트 메시지 전송 결과
    """
    try:
        # logger.info(f"Sending direct message to user: {user_id}")
        result = await slack_client.send_direct_message(user_id, text)
        
        return {
            "success": True,
            "message": "다이렉트 메시지가 성공적으로 전송되었습니다.",
            "data": {
                "user_id": user_id,
                "channel": result.get("channel"),
                "timestamp": result.get("ts"),
                "message_text": text
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in send_direct_message: {e}")
        return {
            "success": False,
            "error": f"다이렉트 메시지 전송 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        # logger.error(f"Unexpected error in send_direct_message: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

# 선택 기능들

@mcp.tool()
async def get_slack_users() -> Dict[str, Any]:
    """
    워크스페이스의 모든 사용자 정보 조회
    
    Returns:
        Dict[str, Any]: 사용자 목록
    """
    try:
        # logger.info("Fetching Slack users")
        users = await slack_client.get_users()
        
        # 사용자 정보 포맷팅 (활성 사용자만)
        formatted_users = []
        for user in users:
            if not user.get("deleted", False) and not user.get("is_bot", False):
                formatted_users.append({
                    "id": user.get("id"),
                    "name": user.get("name"),
                    "real_name": user.get("real_name"),
                    "display_name": user.get("profile", {}).get("display_name", ""),
                    "email": user.get("profile", {}).get("email", ""),
                    "is_admin": user.get("is_admin", False),
                    "is_owner": user.get("is_owner", False),
                    "timezone": user.get("tz", ""),
                    "status": user.get("profile", {}).get("status_text", "")
                })
        
        return {
            "success": True,
            "message": f"총 {len(formatted_users)}명의 사용자를 찾았습니다.",
            "data": {
                "user_count": len(formatted_users),
                "users": formatted_users
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in get_users: {e}")
        return {
            "success": False,
            "error": f"사용자 목록 조회 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        # logger.error(f"Unexpected error in get_users: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

@mcp.tool()
async def add_slack_reaction(channel: str, timestamp: str, name: str) -> Dict[str, Any]:
    """
    특정 메시지에 이모지 반응 추가
    
    Args:
        channel (str): 채널 ID
        timestamp (str): 메시지 타임스탬프
        name (str): 이모지 이름 (콜론 없이, 예: 'thumbsup')
        
    Returns:
        Dict[str, Any]: 반응 추가 결과
    """
    try:
        # logger.info(f"Adding reaction {name} to message {timestamp} in channel {channel}")
        result = await slack_client.add_reaction(channel, timestamp, name)
        
        return {
            "success": True,
            "message": f"반응 '{name}'이 성공적으로 추가되었습니다.",
            "data": {
                "channel": channel,
                "timestamp": timestamp,
                "reaction": name
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in add_reaction: {e}")
        return {
            "success": False,
            "error": f"반응 추가 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        logger.error(f"Unexpected error in add_reaction: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

@mcp.tool()
async def search_slack_messages(query: str, count: int = 20) -> Dict[str, Any]:
    """
    키워드를 통한 메시지 검색
    
    Args:
        query (str): 검색 키워드
        count (int): 반환할 결과 수 (기본값: 20, 최대: 100)
        
    Returns:
        Dict[str, Any]: 검색 결과
    """
    try:
        # count 값 검증 및 조정
        count = max(1, min(count, 100))
        
        # logger.info(f"Searching messages with query: '{query}', count: {count}")
        results = await slack_client.search_messages(query, count)
        
        # 검색 결과 포맷팅
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result.get("text", ""),
                "user": result.get("user"),
                "username": result.get("username"),
                "timestamp": result.get("ts"),
                "channel_id": result.get("channel", {}).get("id"),
                "channel_name": result.get("channel", {}).get("name"),
                "permalink": result.get("permalink")
            })
        
        return {
            "success": True,
            "message": f"검색어 '{query}'에 대해 {len(formatted_results)}개의 결과를 찾았습니다.",
            "data": {
                "query": query,
                "result_count": len(formatted_results),
                "results": formatted_results
            }
        }
        
    except SlackAPIError as e:
        # logger.error(f"Slack API error in search_messages: {e}")
        return {
            "success": False,
            "error": f"메시지 검색 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        # logger.error(f"Unexpected error in search_messages: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

@mcp.tool()
async def test_slack_connection() -> Dict[str, Any]:
    """
    Slack API 연결 테스트
    
    Returns:
        Dict[str, Any]: 연결 테스트 결과
    """
    try:
        # logger.info("Testing Slack API connection")
        result = await slack_client.test_connection()
        
        return {
            "success": True,
            "message": "Slack API 연결이 성공적으로 확인되었습니다.",
            "data": {
                "user": result.get("user"),
                "user_id": result.get("user_id"),
                "team": result.get("team"),
                "team_id": result.get("team_id"),
                "url": result.get("url")
            }
        }
        
    except SlackAPIError as e:
        logger.error(f"Slack API error in test_connection: {e}")
        return {
            "success": False,
            "error": f"연결 테스트 실패: {e.error}",
            "details": e.response
        }
    except Exception as e:
        logger.error(f"Unexpected error in test_connection: {e}")
        return {
            "success": False,
            "error": f"예상치 못한 오류: {str(e)}"
        }

async def cleanup():
    """
    서버 종료 시 정리 작업
    """
    # logger.info("Cleaning up Slack MCP server...")
    await cleanup_slack_client()

def main():
    """
    메인 실행 함수
    """
    try:
        # logger.info("Starting Slack MCP Server...")
        # logger.info(f"Slack Bot Token: {SLACK_BOT_TOKEN[:12]}...")
        
        # stdio transport를 사용하여 MCP 서버 실행
        mcp.run(transport="stdio")
        
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)
    finally:
        # 정리 작업 실행
        asyncio.run(cleanup())

if __name__ == "__main__":
    main() 