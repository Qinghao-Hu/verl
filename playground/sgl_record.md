(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
(AsyncSglangServer pid=4185700)     raise exc
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
(AsyncSglangServer pid=4185700)     await app(scope, receive, sender)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 714, in __call__
(AsyncSglangServer pid=4185700)     await self.middleware_stack(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 734, in app
(AsyncSglangServer pid=4185700)     await route.handle(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 288, in handle
(AsyncSglangServer pid=4185700)     await self.app(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 76, in app
(AsyncSglangServer pid=4185700)     await wrap_app_handling_exceptions(app, request)(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
(AsyncSglangServer pid=4185700)     raise exc
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
(AsyncSglangServer pid=4185700)     await app(scope, receive, sender)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 73, in app
(AsyncSglangServer pid=4185700)     response = await f(request)
(AsyncSglangServer pid=4185700)                ^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/routing.py", line 301, in app
(AsyncSglangServer pid=4185700)     raw_response = await run_endpoint_function(
(AsyncSglangServer pid=4185700)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
(AsyncSglangServer pid=4185700)     return await dependant.call(**values)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/util/tracing/tracing_helper.py", line 495, in _resume_span
(AsyncSglangServer pid=4185700)     return await method(self, *_args, **_kwargs)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/sglang_rollout/async_sglang_server.py", line 80, in chat_completion
(AsyncSglangServer pid=4185700)     [outputs] = await asyncio.gather(output_future)
(AsyncSglangServer pid=4185700)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/asyncio/tasks.py", line 684, in _wrap_awaitable
(AsyncSglangServer pid=4185700)     return await awaitable
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700) ray.exceptions.RayTaskError: ray::WorkerDict.chat_completion() (pid=4174716, ip=10.1.200.9, actor_id=fb9af87e0cc5bb63e969814101000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x15232bbd54c0>)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(AsyncSglangServer pid=4185700)     return self.__get_result()
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(AsyncSglangServer pid=4185700)     raise self._exception
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/single_controller/ray/base.py", line 667, in async_func
(AsyncSglangServer pid=4185700)     return await getattr(self.worker_dict[key], name)(*args, **kwargs)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/single_controller/base/decorator.py", line 546, in async_inner
(AsyncSglangServer pid=4185700)     return await func(*args, **kwargs)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/workers/fsdp_workers.py", line 1502, in chat_completion
(AsyncSglangServer pid=4185700)     ret = await self.rollout.chat_completion(json_request)
(AsyncSglangServer pid=4185700)           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/sglang_rollout/sglang_rollout.py", line 1083, in chat_completion
(AsyncSglangServer pid=4185700)     req = AsyncRolloutRequest(
(AsyncSglangServer pid=4185700)           ^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/pydantic/main.py", line 253, in __init__
(AsyncSglangServer pid=4185700)     validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
(AsyncSglangServer pid=4185700)                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700) pydantic_core._pydantic_core.ValidationError: 1 validation error for AsyncRolloutRequest
(AsyncSglangServer pid=4185700)   Value error, max_prompt_len is required for AsyncRolloutRequest initialization [type=value_error, input_value={'request_id': '34e14ca4-..., 'max_model_len': 1536}, input_type=dict]
(AsyncSglangServer pid=4185700)     For further information visit https://errors.pydantic.dev/2.11/v/value_error
(AsyncSglangServer pid=4185700) ERROR:    Exception in ASGI application
(AsyncSglangServer pid=4185700) Traceback (most recent call last):
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi
(AsyncSglangServer pid=4185700)     result = await app(  # type: ignore[func-returns-value]
(AsyncSglangServer pid=4185700)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
(AsyncSglangServer pid=4185700)     return await self.app(scope, receive, send)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/applications.py", line 1054, in __call__
(AsyncSglangServer pid=4185700)     await super().__call__(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/applications.py", line 112, in __call__
(AsyncSglangServer pid=4185700)     await self.middleware_stack(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/middleware/errors.py", line 187, in __call__
(AsyncSglangServer pid=4185700)     raise exc
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/middleware/errors.py", line 165, in __call__
(AsyncSglangServer pid=4185700)     await self.app(scope, receive, _send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/middleware/exceptions.py", line 62, in __call__
(AsyncSglangServer pid=4185700)     await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
(AsyncSglangServer pid=4185700)     raise exc
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
(AsyncSglangServer pid=4185700)     await app(scope, receive, sender)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 714, in __call__
(AsyncSglangServer pid=4185700)     await self.middleware_stack(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 734, in app
(AsyncSglangServer pid=4185700)     await route.handle(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 288, in handle
(AsyncSglangServer pid=4185700)     await self.app(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 76, in app
(AsyncSglangServer pid=4185700)     await wrap_app_handling_exceptions(app, request)(scope, receive, send)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 53, in wrapped_app
(AsyncSglangServer pid=4185700)     raise exc
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app
(AsyncSglangServer pid=4185700)     await app(scope, receive, sender)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 73, in app
(AsyncSglangServer pid=4185700)     response = await f(request)
(AsyncSglangServer pid=4185700)                ^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/routing.py", line 301, in app
(AsyncSglangServer pid=4185700)     raw_response = await run_endpoint_function(
(AsyncSglangServer pid=4185700)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/routing.py", line 212, in run_endpoint_function
(AsyncSglangServer pid=4185700)     return await dependant.call(**values)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/util/tracing/tracing_helper.py", line 495, in _resume_span
(AsyncSglangServer pid=4185700)     return await method(self, *_args, **_kwargs)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/sglang_rollout/async_sglang_server.py", line 80, in chat_completion
(AsyncSglangServer pid=4185700)     [outputs] = await asyncio.gather(output_future)
(AsyncSglangServer pid=4185700)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/asyncio/tasks.py", line 684, in _wrap_awaitable
(AsyncSglangServer pid=4185700)     return await awaitable
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700) ray.exceptions.RayTaskError: ray::WorkerDict.chat_completion() (pid=4174716, ip=10.1.200.9, actor_id=fb9af87e0cc5bb63e969814101000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x15232bbd54c0>)
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 449, in result
(AsyncSglangServer pid=4185700)     return self.__get_result()
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
(AsyncSglangServer pid=4185700)     raise self._exception
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/single_controller/ray/base.py", line 667, in async_func
(AsyncSglangServer pid=4185700)     return await getattr(self.worker_dict[key], name)(*args, **kwargs)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/single_controller/base/decorator.py", line 546, in async_inner
(AsyncSglangServer pid=4185700)     return await func(*args, **kwargs)
(AsyncSglangServer pid=4185700)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/workers/fsdp_workers.py", line 1502, in chat_completion
(AsyncSglangServer pid=4185700)     ret = await self.rollout.chat_completion(json_request)
(AsyncSglangServer pid=4185700)           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/sglang_rollout/sglang_rollout.py", line 1083, in chat_completion
(AsyncSglangServer pid=4185700)     req = AsyncRolloutRequest(
(AsyncSglangServer pid=4185700)           ^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/pydantic/main.py", line 253, in __init__
(AsyncSglangServer pid=4185700)     validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
(AsyncSglangServer pid=4185700)                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
(AsyncSglangServer pid=4185700) pydantic_core._pydantic_core.ValidationError: 1 validation error for AsyncRolloutRequest
(AsyncSglangServer pid=4185700)   Value error, max_prompt_len is required for AsyncRolloutRequest initialization [type=value_error, input_value={'request_id': 'ba46f275-..., 'max_model_len': 1536}, input_type=dict]
(AsyncSglangServer pid=4185700)     For further information visit https://errors.pydantic.dev/2.11/v/value_error
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,385:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,385:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,386:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,387:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,387:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,421:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,422:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,423:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,505:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,523:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,523:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,523:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,523:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,527:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,527:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,527:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,529:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,529:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,530:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,530:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,530:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,530:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,542:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,542:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,542:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,542:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,545:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,545:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,546:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,607:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,608:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,618:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,618:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,618:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,731:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,731:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,731:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,731:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,731:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,731:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,772:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,773:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,773:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,794:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,794:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,794:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,794:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,794:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,794:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,828:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,828:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,849:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,849:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,849:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,849:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,849:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,850:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,920:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,974:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,974:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,974:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,974:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,974:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:43,974:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,011:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,036:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,036:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,056:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,056:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,056:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,081:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,081:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,082:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,082:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,117:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,117:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,139:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,139:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,156:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,156:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,171:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,172:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,172:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,172:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,172:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,172:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,203:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,203:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,203:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,204:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,241:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,241:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,241:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,242:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,242:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,242:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,242:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,262:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:43003/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,272:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,277:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,277:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,293:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,294:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,294:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,294:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,317:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,317:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,320:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,325:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,325:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,325:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,325:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,328:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,329:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,330:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,347:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,347:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,347:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,347:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,347:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,348:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,348:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,348:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,348:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,348:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,348:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:45563/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,376:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:47279/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,399:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,399:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,431:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,432:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,432:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,432:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,433:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,433:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,496:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,533:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,533:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,556:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,597:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,606:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,624:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,653:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,669:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,682:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,690:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,713:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,714:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,739:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,754:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,774:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,783:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,792:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,813:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,827:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,836:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,858:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,863:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,905:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,919:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,919:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,969:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,969:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
(TaskRunner pid=4168169) ERROR:2025-06-19 14:44:44,990:chat completion failed with exception: 500, message='Attempt to decode JSON with unexpected mimetype: text/plain; charset=utf-8', url='http://10.1.200.9:56211/v1/chat/completions'
(TaskRunner pid=4168169) NoneType: None
Error executing job with overrides: ['ray_init.num_cpus=96', 'algorithm.adv_estimator=grpo', 'data.train_files=/nobackup/qinghao/dataset/reasoning/gsm8k/train.parquet', 'data.val_files=/nobackup/qinghao/dataset/reasoning/gsm8k/test.parquet', 'data.train_batch_size=128', 'data.max_prompt_length=512', 'data.max_response_length=1024', 'data.return_raw_chat=True', 'data.filter_overlong_prompts=True', 'data.truncation=error', 'actor_rollout_ref.model.path=/local/model/qwen3/Qwen3-0.6B', 'actor_rollout_ref.actor.optim.lr=1e-6', 'actor_rollout_ref.model.use_remove_padding=True', 'actor_rollout_ref.actor.ppo_mini_batch_size=128', 'actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16', 'actor_rollout_ref.actor.use_kl_loss=True', 'actor_rollout_ref.actor.kl_loss_coef=0.001', 'actor_rollout_ref.actor.kl_loss_type=low_var_kl', 'actor_rollout_ref.actor.entropy_coeff=0', 'actor_rollout_ref.model.enable_gradient_checkpointing=True', 'actor_rollout_ref.actor.fsdp_config.param_offload=False', 'actor_rollout_ref.actor.fsdp_config.optimizer_offload=False', 'actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.rollout.tensor_model_parallel_size=2', 'actor_rollout_ref.rollout.name=sglang', 'actor_rollout_ref.rollout.mode=async', 'actor_rollout_ref.rollout.gpu_memory_utilization=0.6', 'actor_rollout_ref.rollout.n=2', 'actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16', 'actor_rollout_ref.ref.fsdp_config.param_offload=True', 'algorithm.use_kl_in_reward=False', 'trainer.critic_warmup=0', 'trainer.val_before_train=False', 'trainer.logger=[console]', 'trainer.project_name=Qwen3-RL', 'trainer.experiment_name=debug-split', 'trainer.n_gpus_per_node=8', 'trainer.nnodes=1', 'trainer.save_freq=20', 'trainer.test_freq=5', 'trainer.total_epochs=1']
Traceback (most recent call last):
  File "/home/qinghao/workdir/verl-dev/verl/trainer/main_ppo.py", line 31, in main
    run_ppo(config)
  File "/home/qinghao/workdir/verl-dev/verl/trainer/main_ppo.py", line 54, in run_ppo
    ray.get(runner.run.remote(config))
  File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 21, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 103, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/_private/worker.py", line 2822, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/_private/worker.py", line 930, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AssertionError): ray::TaskRunner.run() (pid=4168169, ip=10.1.200.9, actor_id=8072a771698f8716fba91e4801000000, repr=<main_ppo.TaskRunner object at 0x15555109eed0>)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/workdir/verl-dev/verl/trainer/main_ppo.py", line 195, in run
    trainer.fit()
  File "/home/qinghao/workdir/verl-dev/verl/trainer/ppo/ray_trainer.py", line 969, in fit
    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/async_server.py", line 228, in generate_sequences
    return future.result()
           ^^^^^^^^^^^^^^^
  File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception
  File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/chat_scheduler.py", line 425, in generate_sequences
    output_batch = self.completion_callback.postprocess(batch, batch_conversations, n=n)
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/chat_scheduler.py", line 180, in postprocess
    response_mask = self._mask_out_tools_calling_tokens(batch.non_tensor_batch["raw_prompt"].repeat(n, axis=0), batch_conversations, responses["input_ids"], responses["attention_mask"])
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/chat_scheduler.py", line 238, in _mask_out_tools_calling_tokens
    assert len(responses) > 0, f"responses is empty: {responses}"
           ^^^^^^^^^^^^^^^^^^
AssertionError: responses is empty: []

Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
(AsyncSglangServer pid=4185699) User exception type <class 'pydantic_core._pydantic_core.ValidationError'> in RayTaskError can't be subclassed! This exception is raised as RayTaskError only. You can use `ray_task_error.cause` to access the user exception. Failure in subclassing: ValidationError.__new__() missing 1 required positional argument: 'line_errors' [repeated 244x across cluster]
(AsyncSglangServer pid=4185699) ERROR:    Exception in ASGI application [repeated 247x across cluster]
(AsyncSglangServer pid=4185699) Traceback (most recent call last): [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/uvicorn/protocols/http/httptools_impl.py", line 409, in run_asgi [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     result = await app(  # type: ignore[func-returns-value] [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 714, in __call__ [repeated 1729x across cluster]
(AsyncSglangServer pid=4185699)     return await self.app(scope, receive, send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     await super().__call__(scope, receive, send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     await self.middleware_stack(scope, receive, send) [repeated 494x across cluster]
(AsyncSglangServer pid=4185699)     raise exc [repeated 741x across cluster]
(AsyncSglangServer pid=4185699)     await self.app(scope, receive, _send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/_exception_handler.py", line 42, in wrapped_app [repeated 988x across cluster]
(AsyncSglangServer pid=4185699)     await app(scope, receive, sender) [repeated 494x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/routing.py", line 301, in app [repeated 988x across cluster]
(AsyncSglangServer pid=4185699)     await route.handle(scope, receive, send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/starlette/routing.py", line 288, in handle [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     await self.app(scope, receive, send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     await wrap_app_handling_exceptions(app, request)(scope, receive, send) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     response = await f(request) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)                ^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     raw_response = await run_endpoint_function( [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/fastapi/routing.py", line 212, in run_endpoint_function [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     return await dependant.call(**values) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/ray/util/tracing/tracing_helper.py", line 495, in _resume_span [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     return await method(self, *_args, **_kwargs) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 494x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/sglang_rollout/async_sglang_server.py", line 80, in chat_completion [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     [outputs] = await asyncio.gather(output_future) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/asyncio/tasks.py", line 684, in _wrap_awaitable [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     return await awaitable [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699) ray.exceptions.RayTaskError: ray::WorkerDict.chat_completion() (pid=4174183, ip=10.1.200.9, actor_id=c000dafc9726fe433cadbed201000000, repr=<verl.single_controller.ray.base.WorkerDict object at 0x1523802395e0>) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 449, in result [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     return self.__get_result() [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/concurrent/futures/_base.py", line 401, in __get_result [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     raise self._exception [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/workdir/verl-dev/verl/single_controller/ray/base.py", line 667, in async_func [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     return await getattr(self.worker_dict[key], name)(*args, **kwargs) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/workdir/verl-dev/verl/single_controller/base/decorator.py", line 546, in async_inner [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     return await func(*args, **kwargs) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)            ^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/workdir/verl-dev/verl/workers/fsdp_workers.py", line 1502, in chat_completion [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     ret = await self.rollout.chat_completion(json_request) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/workdir/verl-dev/verl/workers/rollout/sglang_rollout/sglang_rollout.py", line 1083, in chat_completion [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     req = AsyncRolloutRequest( [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)           ^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   File "/home/qinghao/miniconda3/envs/dev/lib/python3.12/site-packages/pydantic/main.py", line 253, in __init__ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self) [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ [repeated 247x across cluster]
(AsyncSglangServer pid=4185699) pydantic_core._pydantic_core.ValidationError: 1 validation error for AsyncRolloutRequest [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)   Value error, max_prompt_len is required for AsyncRolloutRequest initialization [type=value_error, input_value={'request_id': '9ebeb7d2-..., 'max_model_len': 1536}, input_type=dict] [repeated 247x across cluster]
(AsyncSglangServer pid=4185699)     For further information visit https://errors.pydantic.dev/2.11/v/value_error [repeated 247x across cluster]
srun: error: dgx-06: task 0: Exited with exit code 1
(dev) qinghao@dgx-01 ➜ ~/workdir/verl-dev (fastrl ✗) $ 