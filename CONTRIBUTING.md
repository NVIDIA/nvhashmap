<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# OSS contribution rules

## Coding guidelines

- Please follow the existing conventions in the relevant file / namespace when you add new code or when you extend/fix existing functionality.

- To maintain consistency in code formatting and style, you should run `clang-format` on the modified sources with the provided configuration file.

- Format individual source files:
  ```shell
  # -style=file : Obtain the formatting rules from .clang-format
  # -i          : In-place modification of the processed file
  clang-format -style=file -fallback-style=none -i <file(s) to process>
  ```

- Format entire codebase (for project maintainers only):
  ```shell
  find benchmark include tests tools -iname *.hpp -o -iname *.cpp \
    | xargs clang-format -style=file -fallback-style=none -i
  ```

- Avoid introducing unnecessary complexity into existing code so that maintainability and readability are preserved.

- Keep pull requests (PRs) as concise as possible:
    - Avoid committing commented-out code.
    - Each PR should address a single concern.
    - If there are several otherwise-unrelated things that should be fixed to reach a desired endpoint, our recommendation is to open several PRs and indicate dependencies in the descriptions.

- Ensure that the build log is clean, meaning no warnings or errors should be present.

- Ensure that all unit tests pass prior to submitting your code.

- All components should have an accompanying test.

- To add or disable functionality:
    - Add a CMake option with a default value that matches the existing behavior.
    - Where entire files can be included/excluded based on the value of this option, selectively include/exclude the relevant files from compilation by modifying `CMakeLists.txt` rather than using `#if` guards around the entire body of each file.
    - Where the functionality involves minor changes to existing files, use `#if` guards.

- Make sure that you can contribute your work to open source (no license and/or patent conflict is introduced by your code). You will need to [`sign`](#signing-your-work) your commit.

- Thanks in advance for your patience as we review your contributions; we do appreciate them!


## Pull requests
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the [upstream](https://github.com/nvidia/nvhashmap) nvHashMap repository.

2. Git clone the forked repository and push changes to your personal fork.
    ```shell
    git clone https://github.com/YOUR_USERNAME/YOUR_FORK.git nvhashmap
    git push -u origin <local-branch>:<remote-branch>
    ```

3. Once the code changes are staged on the fork and ready for review, initiate a [Pull Request](https://help.github.com/en/articles/creating-a-pull-request) (PR) to merge the changes from a branch of your fork into a selected branch of *upstream*.
    * If your PR relates to an issue, please link the issue in the description of your PR.
    * Exercise caution when selecting the source and target branches for the PR.
    * Creation of a PR kicks off the code review process.
    * While under review, mark your PRs as work-in-progress by prefixing the PR title with [WIP].

4. Since there is no CI/CD process in place yet, the PR will be accepted and thereby resolved issues closed only after adequate testing has been manually completed by us.


## Signing your work

* We require that all contributors *sign-off* on their commits. This certifies that the contribution is your original work, or you have the rights to submit it under the same license, or a compatible license.

* Contributions which contain commits that are not signed-off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```shell
  git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO:
  ```
  Developer Certificate of Origin
  Version 1.1

  Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

  Everyone is permitted to copy and distribute verbatim copies of this
  license document, but changing it is not allowed.


  Developer's Certificate of Origin 1.1

  By making a contribution to this project, I certify that:

  (a) The contribution was created in whole or in part by me and I
      have the right to submit it under the open source license
      indicated in the file; or

  (b) The contribution is based upon previous work that, to the best
      of my knowledge, is covered under an appropriate open source
      license and I have the right under that license to submit that
      work with modifications, whether created in whole or in part
      by me, under the same open source license (unless I am
      permitted to submit under a different license), as indicated
      in the file; or

  (c) The contribution was provided directly to me by some other
      person who certified (a), (b) or (c) and I have not modified
      it.

  (d) I understand and agree that this project and the contribution
      are public and that a record of the contribution (including all
      personal information I submit with it, including my sign-off) is
      maintained indefinitely and may be redistributed consistent with
      this project or the open source license(s) involved.
  ```
