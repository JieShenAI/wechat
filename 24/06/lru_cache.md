# Lou_cache vs cache

在Python中，`lru_cache`和`cache`都是`functools`模块提供的装饰器，用于缓存函数的结果，但它们的功能和使用场景略有不同。

### `functools.lru_cache`

`lru_cache`表示“最近最少使用”缓存。它是一个装饰器，用于缓存函数调用的结果。当缓存达到设定的最大容量时，会丢弃最近最少使用的缓存项。这对于一些计算量大且频繁调用的函数非常有用。

- **语法**：`@functools.lru_cache(maxsize=128, typed=False)`
- **参数**：
  - `maxsize`：指定缓存的最大容量。如果设置为`None`，缓存大小不受限制。
  - `typed`：如果设置为`True`，则会将不同类型的参数视为不同的调用（例如，`f(3)`和`f(3.0)`会分别缓存）。

- **示例**：
  ```python
  from functools import lru_cache
  
  @lru_cache(maxsize=100)
  def expensive_function(x):
      # 模拟耗时计算
      return x * x
  ```

### `functools.cache`

`cache`是一个更简单版本的缓存装饰器。它是`lru_cache(maxsize=None)`的别名，表示提供一个不受限制的缓存。这在需要缓存所有函数调用结果且不考虑缓存淘汰策略时非常有用。

- **语法**：`@functools.cache`
- **示例**：
  ```python
  from functools import cache
  
  @cache
  def expensive_function(x):
      # 模拟耗时计算
      return x * x
  ```

### 关键区别

1. **淘汰策略**：
   - `lru_cache`：使用最近最少使用的淘汰策略。当缓存达到最大容量时，丢弃最近最少使用的项。
   - `cache`：没有淘汰策略，缓存项数量不受限制。

2. **定制化**：
   - `lru_cache`：允许设置缓存大小（`maxsize`）和类型敏感性（`typed`）。
   - `cache`：没有定制化选项，相当于`lru_cache(maxsize=None)`。

### 使用场景

- **`lru_cache`**：适用于需要限制内存使用且对使用顺序敏感的缓存场景。
- **`cache`**：适用于需要简单且不受限制的缓存场景。



## leetcode

题目：https://leetcode.cn/problems/special-permutations/description/

我讲下述代码提交后，发现 `@lru_cache(maxsize=None)` 时间超时，` @cache` 能够成功提交；

> 建议大家leetcode 刷题的时候，还是使用 @cache 好一点，简单无脑还快一点；

```python
class Solution:
    def specialPerm(self, nums: List[int]) -> int:

        @cache
        # @lru_cache(maxsize=None)
        def dfs(rear:int, lefts: tuple):
            if len(lefts) == 0:
                return 1

            res = 0
            for item in lefts:
                if item % rear == 0 or rear % item == 0:
                    res += dfs(item, tuple(set(lefts) - set([item])))
            
            return res % (1e9 + 7)
            

        res = 0
        for item in nums:
            res += dfs(
                    item, 
                    tuple(set(nums) - set([item]))
                )

        return int(res % (1e9 + 7))
```

