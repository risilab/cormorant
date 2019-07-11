from cormorant.cg_lib.gen_cg import _GenCG as gen_cg_cpp
from cormorant.cg_lib.gen_cg_py import gen_cg as gen_cg_py

maxl = 5

cg_cpp = gen_cg_cpp.gen_cg_coefffs(maxl)
cg_py = gen_cg_py(maxl)

# print(cg_cpp)
# print(cg_py)

for cpp, py in zip(cg_cpp, cg_py):
    print(cpp.shape, py.shape)
    print((cpp-py).abs().max())

breakpoint()
