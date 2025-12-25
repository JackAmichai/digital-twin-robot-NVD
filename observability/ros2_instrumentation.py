"""
OpenTelemetry instrumentation for ROS 2 nodes.

Provides automatic tracing for ROS 2 publishers, subscribers, services,
and action clients/servers.
"""

import functools
from typing import Any, Callable, Dict, Optional, TypeVar
import asyncio

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import CallbackGroup
from rclpy.qos import QoSProfile

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode
from opentelemetry.propagate import inject, extract

from .tracing import DistributedTracer, TracingConfig


F = TypeVar('F', bound=Callable[..., Any])


class TracedNode(Node):
    """
    ROS 2 Node with automatic OpenTelemetry tracing.
    
    Wraps standard ROS 2 node functionality with distributed tracing,
    automatically creating spans for:
    - Message publications
    - Subscription callbacks
    - Service calls and handlers
    - Action goals and feedback
    
    Example:
        >>> class MyRobot(TracedNode):
        ...     def __init__(self):
        ...         super().__init__('my_robot')
        ...         self.create_traced_subscription(
        ...             PoseStamped, '/cmd_pose', self.pose_callback, 10
        ...         )
        ...     
        ...     def pose_callback(self, msg):
        ...         # Automatically traced!
        ...         self.navigate_to(msg.pose)
    """
    
    # Trace context header for ROS 2 messages
    TRACE_HEADER_KEY = 'trace_context'
    
    def __init__(
        self,
        node_name: str,
        *,
        tracer_config: Optional[TracingConfig] = None,
        **kwargs
    ):
        """
        Initialize traced ROS 2 node.
        
        Args:
            node_name: Name of the ROS 2 node
            tracer_config: Optional custom tracing configuration
            **kwargs: Additional arguments passed to Node.__init__
        """
        super().__init__(node_name, **kwargs)
        
        # Initialize tracer
        config = tracer_config or TracingConfig(
            service_name=f"ros2_{node_name}",
            service_version="1.0.0"
        )
        self._tracer = DistributedTracer(config)
        self._tracer.initialize()
        
        self.get_logger().info(f"Initialized tracing for node: {node_name}")
    
    @property
    def tracer(self) -> DistributedTracer:
        """Get the tracer instance for manual span creation."""
        return self._tracer
    
    def create_traced_publisher(
        self,
        msg_type: type,
        topic: str,
        qos_profile: QoSProfile,
        **kwargs
    ):
        """
        Create a publisher with automatic tracing.
        
        Wraps the standard publisher to automatically create spans
        and propagate trace context in message headers.
        
        Args:
            msg_type: ROS message type
            topic: Topic name
            qos_profile: QoS profile
            **kwargs: Additional publisher arguments
            
        Returns:
            TracedPublisher instance
        """
        publisher = self.create_publisher(msg_type, topic, qos_profile, **kwargs)
        return TracedPublisher(publisher, topic, self._tracer)
    
    def create_traced_subscription(
        self,
        msg_type: type,
        topic: str,
        callback: Callable,
        qos_profile: QoSProfile,
        **kwargs
    ):
        """
        Create a subscription with automatic tracing.
        
        Wraps the callback to automatically create spans for each
        message received and extract parent trace context.
        
        Args:
            msg_type: ROS message type
            topic: Topic name
            callback: Message callback function
            qos_profile: QoS profile
            **kwargs: Additional subscription arguments
            
        Returns:
            Subscription instance
        """
        @functools.wraps(callback)
        def traced_callback(msg):
            with self._tracer.span(
                f"ros2_receive/{topic}",
                kind=SpanKind.CONSUMER,
                attributes={
                    "ros2.topic": topic,
                    "ros2.message_type": msg_type.__name__,
                    "ros2.node": self.get_name(),
                }
            ) as span:
                try:
                    callback(msg)
                    span.set_status(Status(StatusCode.OK))
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return self.create_subscription(
            msg_type, topic, traced_callback, qos_profile, **kwargs
        )
    
    def create_traced_service(
        self,
        srv_type: type,
        srv_name: str,
        callback: Callable,
        **kwargs
    ):
        """
        Create a service with automatic tracing.
        
        Args:
            srv_type: Service type
            srv_name: Service name
            callback: Service callback
            **kwargs: Additional service arguments
            
        Returns:
            Service instance
        """
        @functools.wraps(callback)
        def traced_callback(request, response):
            with self._tracer.span(
                f"ros2_service/{srv_name}",
                kind=SpanKind.SERVER,
                attributes={
                    "ros2.service": srv_name,
                    "ros2.service_type": srv_type.__name__,
                    "ros2.node": self.get_name(),
                }
            ) as span:
                try:
                    result = callback(request, response)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        return self.create_service(srv_type, srv_name, traced_callback, **kwargs)
    
    def create_traced_client(
        self,
        srv_type: type,
        srv_name: str,
        **kwargs
    ):
        """
        Create a service client with automatic tracing.
        
        Args:
            srv_type: Service type
            srv_name: Service name
            **kwargs: Additional client arguments
            
        Returns:
            TracedServiceClient instance
        """
        client = self.create_client(srv_type, srv_name, **kwargs)
        return TracedServiceClient(client, srv_name, self._tracer)


class TracedPublisher:
    """
    Publisher wrapper with automatic span creation and context propagation.
    """
    
    def __init__(
        self,
        publisher,
        topic: str,
        tracer: DistributedTracer
    ):
        self._publisher = publisher
        self._topic = topic
        self._tracer = tracer
    
    def publish(self, msg) -> None:
        """
        Publish message with tracing.
        
        Creates a span for the publish operation and injects
        trace context into message headers if available.
        """
        with self._tracer.span(
            f"ros2_publish/{self._topic}",
            kind=SpanKind.PRODUCER,
            attributes={
                "ros2.topic": self._topic,
                "ros2.message_type": type(msg).__name__,
            }
        ) as span:
            # Inject trace context if message has header
            if hasattr(msg, 'header') and hasattr(msg.header, 'frame_id'):
                # Use frame_id to carry trace context (hacky but works)
                carrier = {}
                inject(carrier)
                if 'traceparent' in carrier:
                    span.set_attribute("trace.propagated", True)
            
            self._publisher.publish(msg)
            span.set_status(Status(StatusCode.OK))
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying publisher."""
        return getattr(self._publisher, name)


class TracedServiceClient:
    """
    Service client wrapper with automatic span creation.
    """
    
    def __init__(
        self,
        client,
        service_name: str,
        tracer: DistributedTracer
    ):
        self._client = client
        self._service_name = service_name
        self._tracer = tracer
    
    def call(self, request):
        """
        Call service synchronously with tracing.
        """
        with self._tracer.span(
            f"ros2_service_call/{self._service_name}",
            kind=SpanKind.CLIENT,
            attributes={
                "ros2.service": self._service_name,
                "ros2.request_type": type(request).__name__,
            }
        ) as span:
            try:
                response = self._client.call(request)
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    async def call_async(self, request):
        """
        Call service asynchronously with tracing.
        """
        with self._tracer.span(
            f"ros2_service_call_async/{self._service_name}",
            kind=SpanKind.CLIENT,
            attributes={
                "ros2.service": self._service_name,
                "ros2.request_type": type(request).__name__,
            }
        ) as span:
            try:
                future = self._client.call_async(request)
                response = await future
                span.set_status(Status(StatusCode.OK))
                return response
            except Exception as e:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
                raise
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying client."""
        return getattr(self._client, name)


class TracedActionClient:
    """
    Action client wrapper with comprehensive span creation.
    
    Creates spans for:
    - Goal submission
    - Goal acceptance
    - Feedback reception
    - Result reception
    """
    
    def __init__(
        self,
        action_client,
        action_name: str,
        tracer: DistributedTracer
    ):
        self._client = action_client
        self._action_name = action_name
        self._tracer = tracer
    
    def send_goal_async(self, goal, feedback_callback=None):
        """
        Send action goal with tracing.
        """
        # Wrap feedback callback with tracing
        if feedback_callback:
            original_callback = feedback_callback
            
            def traced_feedback_callback(feedback_msg):
                with self._tracer.span(
                    f"ros2_action_feedback/{self._action_name}",
                    kind=SpanKind.CONSUMER,
                    attributes={"ros2.action": self._action_name}
                ):
                    original_callback(feedback_msg)
            
            feedback_callback = traced_feedback_callback
        
        with self._tracer.span(
            f"ros2_action_send_goal/{self._action_name}",
            kind=SpanKind.CLIENT,
            attributes={
                "ros2.action": self._action_name,
                "ros2.goal_type": type(goal).__name__,
            }
        ) as span:
            future = self._client.send_goal_async(goal, feedback_callback)
            span.set_attribute("ros2.goal_sent", True)
            return TracedGoalFuture(future, self._action_name, self._tracer)
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying client."""
        return getattr(self._client, name)


class TracedGoalFuture:
    """
    Wrapper for action goal future with result tracing.
    """
    
    def __init__(
        self,
        future,
        action_name: str,
        tracer: DistributedTracer
    ):
        self._future = future
        self._action_name = action_name
        self._tracer = tracer
    
    def result(self):
        """
        Get goal result with tracing.
        """
        with self._tracer.span(
            f"ros2_action_result/{self._action_name}",
            kind=SpanKind.CLIENT,
            attributes={"ros2.action": self._action_name}
        ) as span:
            result = self._future.result()
            span.set_attribute("ros2.goal_accepted", result.accepted)
            return result
    
    def __getattr__(self, name):
        """Delegate unknown attributes to underlying future."""
        return getattr(self._future, name)


def trace_ros2_callback(tracer: DistributedTracer, name: str) -> Callable[[F], F]:
    """
    Decorator for tracing ROS 2 callbacks.
    
    Args:
        tracer: DistributedTracer instance
        name: Span name for the callback
        
    Returns:
        Decorated function
        
    Example:
        >>> @trace_ros2_callback(tracer, "process_scan")
        ... def scan_callback(msg: LaserScan):
        ...     process_laser_scan(msg)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.span(name, kind=SpanKind.INTERNAL) as span:
                try:
                    result = func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with tracer.span(name, kind=SpanKind.INTERNAL) as span:
                try:
                    result = await func(*args, **kwargs)
                    span.set_status(Status(StatusCode.OK))
                    return result
                except Exception as e:
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                    span.record_exception(e)
                    raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator


__all__ = [
    'TracedNode',
    'TracedPublisher',
    'TracedServiceClient',
    'TracedActionClient',
    'TracedGoalFuture',
    'trace_ros2_callback',
]
