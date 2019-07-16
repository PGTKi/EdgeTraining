# 关于 dynamic graph 在 TVM 中实现的调查。

## Relay 介绍

* Relay 是用编程语言的角度设计的上层 IR
* Relay 的特点包括 (1) Rich type system (类型检查, 类型推断, template, 多态, etc) (2) 借鉴了 Programming Language 里的各种设计，方便用各种经典的编译器优化。

