# Third-party notices

Semigroup-JEPA includes or adapts the following components. Project-owned code
is distributed under the repository's MIT license; the terms below continue to
apply to their respective upstream components.

## DINO-WM

The optional DINO-WM predictor is adapted from
[`gaoyuezhou/dino_wm`](https://github.com/gaoyuezhou/dino_wm) at commit
`0a9492fa12044b852ae9e001cc74604b79c8bb0c`, copyright (c) 2025 gaoyuezhou,
under the MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## stable-pretraining

The historical training code used
[`rbalestr-lab/stable-pretraining`](https://github.com/rbalestr-lab/stable-pretraining)
at commit `3b7706dc83c2f4c16d2bed4b64e6f29273c197c8`. Required encoder,
SIGReg, and optimizer contracts were ported into this repository. The upstream
project is MIT licensed, copyright (c) 2024 rbalestr-lab, under the same MIT
terms reproduced above.

## DINOv2

DINOv2 source and weights are not redistributed here. The code pins
[`facebookresearch/dinov2`](https://github.com/facebookresearch/dinov2) to
commit `85a24602099d397264d5b30461ad7f3bfd726ca1` and verifies the upstream
ViT-S/14 weight payload by SHA-256. The upstream DINOv2 source and model-weight
terms apply.

## MuJoCo Menagerie

The generators fetch only `unitree_z1` and `franka_emika_panda` from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie)
at commit `c1a4eeb85694ae1dffe33ff1797d4e528928a133`. Assets are not vendored;
their tree hashes are checked and the upstream `LICENSE` remains beside each
provisioned model. MuJoCo Menagerie is Apache-2.0 licensed, and individual
model directories may contain additional attribution information.
