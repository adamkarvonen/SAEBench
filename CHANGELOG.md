# CHANGELOG

## v0.5.0 (2025-10-19)

### Chore

* chore: update semantic-release action version ([`b899b37`](https://github.com/adamkarvonen/SAEBench/commit/b899b3784346c996517dab6022b4f34850bda1f0))

### Feature

* feat: adding options to absorption eval to improve running speed ([`485ef87`](https://github.com/adamkarvonen/SAEBench/commit/485ef87d76eb04b38d4f637e3da6f47d6c47729d))

### Unknown

* Merge pull request #82 from chanind/update-semantic-release-action

chore: update semantic-release action version ([`03f24ee`](https://github.com/adamkarvonen/SAEBench/commit/03f24ee0e1d7315470602d6f1a9b0cffa4cf0117))

* Merge pull request #81 from chanind/absorption-perf

feat: adding options to absorption eval to improve running speed ([`2058518`](https://github.com/adamkarvonen/SAEBench/commit/20585181bfe829f379521a51540fdd9ee3be36b0))

* bust venv cache ([`d752dd8`](https://github.com/adamkarvonen/SAEBench/commit/d752dd885aea2d73ff93fbca43acc56f4460b3b7))

* Merge pull request #80 from chanind/saelens-v6-compat

feat: updating to SAELens v6 ([`5de6c00`](https://github.com/adamkarvonen/SAEBench/commit/5de6c001dffc0d56c65cd4f888224c04e809c20d))

* adding a  method to custom SAE cfg to mimic SAELens ([`f837256`](https://github.com/adamkarvonen/SAEBench/commit/f8372565aad1ff4df99d619181c22ccee4d11bcb))

* use a helper method to smooth over differences in converting custom SAE cfg and SAELens cfg to dict ([`1b21c23`](https://github.com/adamkarvonen/SAEBench/commit/1b21c2327051fbbbb950dc943a60455df0cf6da6))

* remove metadata from sae cfg just in case ([`fa79666`](https://github.com/adamkarvonen/SAEBench/commit/fa79666bd31e1b255714a66ed3fe0892035a9298))

* revert acceptance test, it was right the first time ([`52e1040`](https://github.com/adamkarvonen/SAEBench/commit/52e10406f6c87856623e60ba3bcff5db61020a6a))

* fixes ([`5f1cf98`](https://github.com/adamkarvonen/SAEBench/commit/5f1cf98112ff789ecee185121ad6d3cf68679466))

* update core e2e test path ([`2951c7d`](https://github.com/adamkarvonen/SAEBench/commit/2951c7d056bfe9e6b76a7f4b8d95950c714fa8af))

* more fixes ([`a22cb8b`](https://github.com/adamkarvonen/SAEBench/commit/a22cb8b732403780fa5300e479b60e244820aa72))

* gracefully handle failure to fold decoder ([`40f65da`](https://github.com/adamkarvonen/SAEBench/commit/40f65da766d11b8d14fad5fde856f054c2f17045))

* standardize SAE configs when loading rather than when calling ([`fd12811`](https://github.com/adamkarvonen/SAEBench/commit/fd12811a2ccbc4d263375e1f99914d8d9ee1c777))

* updating to SAELens v6 ([`1fad11f`](https://github.com/adamkarvonen/SAEBench/commit/1fad11f0b802e64fcb5bdbfe846f0a07d24ef9b8))

* Pin dependencies ([`7f61449`](https://github.com/adamkarvonen/SAEBench/commit/7f61449e015a0fade7a88d4d3cc3790edd550efc))

## v0.4.2 (2025-07-15)

### Fix

* fix: pinning saelens to v5.x.x ([`14bedb5`](https://github.com/adamkarvonen/SAEBench/commit/14bedb5e677ff3edd2df48eda7a9ddaf1b4a5b5d))

### Unknown

* Merge pull request #75 from chanind/pin-saelens

fix: pinning saelens to v5.x.x ([`d7e3b15`](https://github.com/adamkarvonen/SAEBench/commit/d7e3b1553142a146754ef433cf7793fc18d1648f))

* fixing typing ([`7d370e2`](https://github.com/adamkarvonen/SAEBench/commit/7d370e281287a8fb0496a905058256d2405990f1))

* Merge pull request #74 from SrGonao/patch-1

Change autointerp reference from the old blogpost to the new article. ([`1a6641b`](https://github.com/adamkarvonen/SAEBench/commit/1a6641ba62207a352f3103bac679c6e202702002))

* Update README.md

Change autointerp reference from the old blogpost to the new article. ([`7247b1c`](https://github.com/adamkarvonen/SAEBench/commit/7247b1cd2efd818f77044dd596d89262e2dbf226))

## v0.4.1 (2025-04-11)

### Chore

* chore: fixing linting and formatting ([`a839843`](https://github.com/adamkarvonen/SAEBench/commit/a83984373f1e557e0d2b2040378b63a6c107208c))

### Fix

* fix: fix tokenization with mistral for absorption benchmark ([`ce3bccc`](https://github.com/adamkarvonen/SAEBench/commit/ce3bccc39316ad2f4d431a38785d5e128df94a9b))

### Unknown

* Merge pull request #72 from chanind/fix-ci

chore: fixing linting and formatting ([`e751b2f`](https://github.com/adamkarvonen/SAEBench/commit/e751b2f9d8c38f355b7a7a693c750652c429a42b))

* Merge pull request #70 from chanind/fix-mistral-tokenization

Fix mistral tokenization ([`5181bcc`](https://github.com/adamkarvonen/SAEBench/commit/5181bcc6c0568c41564227c43521cf323f894260))

* prepend newline to all examples for mistral ([`f7b1ad0`](https://github.com/adamkarvonen/SAEBench/commit/f7b1ad043a5bad7d40b94e55b9f68d50781f4084))

* Add documentation ([`0fbbe25`](https://github.com/adamkarvonen/SAEBench/commit/0fbbe2564ef8612f895c7acda82105123a0f3a52))

* Merge pull request #67 from demiantill/absorption-valid-feature-threshold

GT probe f1 score threshold to check valid features for absorption calculation ([`de26d33`](https://github.com/adamkarvonen/SAEBench/commit/de26d33d600fa60117f503332e8eab85132ead5a))

* added GT probe f1 score threshold to be considered a valid feature for use in absorption evaluation ([`996daa0`](https://github.com/adamkarvonen/SAEBench/commit/996daa0deaf57faa8f57662714d9247264a16ca5))

* Merge pull request #62 from demiantill/absorption-fraction-thresholds

Support for using thresholds with the absorption fraction metric ([`2c574e5`](https://github.com/adamkarvonen/SAEBench/commit/2c574e54ef3885635db667f7f85066c98a4dd047))

* updated variable name ([`8271ab0`](https://github.com/adamkarvonen/SAEBench/commit/8271ab08d8850aeb99ae8d0e1dbdb9215485a813))

* added support for using thresholds with the absorption fraction metric ([`ca9d806`](https://github.com/adamkarvonen/SAEBench/commit/ca9d806aac04e7e54f44cef92401eb4771e0581a))

* Merge pull request #66 from tatsuropfgt/fix/typos

Fix error messages ([`c5c90a1`](https://github.com/adamkarvonen/SAEBench/commit/c5c90a136c8db8e704e401328a49b19932ad56c9))

* Fix error messages ([`eb0e21e`](https://github.com/adamkarvonen/SAEBench/commit/eb0e21e6483af451275d6a407ff7013ae5790c41))

* Proceed to next evaluation if exception encountered ([`c320790`](https://github.com/adamkarvonen/SAEBench/commit/c320790319a3ae278f86ee666341fad18abbc6a1))

* Various improvements to reduce RAVEL memory fragmentation ([`07c9238`](https://github.com/adamkarvonen/SAEBench/commit/07c9238f7343cbafaf2de442c15de6506e018941))

* Merge pull request #65 from alowet/main

Fix autointerp demo.py ([`db5a2d6`](https://github.com/adamkarvonen/SAEBench/commit/db5a2d6573c5f5adeeb128f8e7fb0f876dc584ca))

* Allow n_latents to be None so that override_latents works. Pass additional arguments to run_eval in demo.py so that it works. ([`421ac62`](https://github.com/adamkarvonen/SAEBench/commit/421ac62d8c435e5e4b72597adb1e5b10b77360ad))

* Merge pull request #64 from adamkarvonen/ravel_types

Support RAVEL JSON Schema / types ([`02585e0`](https://github.com/adamkarvonen/SAEBench/commit/02585e0f9af3746b92e05e028f12b62ab05f66a9))

* Support RAVEL JSON Schema / types ([`c8eab86`](https://github.com/adamkarvonen/SAEBench/commit/c8eab86089ff2aaeab0453b6734df6115a767130))

* Merge pull request #63 from adamkarvonen/ravel_pytorch

Implement RAVEL ([`ccd7bfb`](https://github.com/adamkarvonen/SAEBench/commit/ccd7bfb7ec396a8b5d752132e9d6aa833c3075f1))

* Move baseline suite to main README ([`e69de67`](https://github.com/adamkarvonen/SAEBench/commit/e69de6762a18fb4a4fd7eb860bc903e0097d923f))

* Rename supervised baseline to mdas ([`228ca11`](https://github.com/adamkarvonen/SAEBench/commit/228ca11e3cc586767dfd659e066f4f2432921b2d))

* Remove old das ([`6574cbe`](https://github.com/adamkarvonen/SAEBench/commit/6574cbede2ffe4502d9f3b0dcbb9eb8bbb11f005))

* Add end to end test for RAVEL ([`fc4ebf1`](https://github.com/adamkarvonen/SAEBench/commit/fc4ebf1734fb665f1774333eda283a4484f004da))

* Various Ravel improvements ([`2875b66`](https://github.com/adamkarvonen/SAEBench/commit/2875b668fbe1f23ca83977e7461798fc7395e6b8))

* Add debugging note ([`5160b81`](https://github.com/adamkarvonen/SAEBench/commit/5160b814922b95d7a80f26daa0c0302e56d039d8))

* standardize batch sizes ([`6db967f`](https://github.com/adamkarvonen/SAEBench/commit/6db967fe62f26e5236517ddbc88f0627b52521dd))

* Fix PCA implementation ([`246f57a`](https://github.com/adamkarvonen/SAEBench/commit/246f57a42b111a1dff6811c5b07be61cceba1fbf))

* Add optional flag for adding SAE error term ([`b39411f`](https://github.com/adamkarvonen/SAEBench/commit/b39411f7bf11eb8023ec96304e281bf2b88afabd))

* More memory usage reduction ([`6207cd4`](https://github.com/adamkarvonen/SAEBench/commit/6207cd45183bae8c9a685c6ee67950bfa912f40b))

* Make sure to use torch.no_grad when appropriate ([`c681fd8`](https://github.com/adamkarvonen/SAEBench/commit/c681fd875b7a7f8c08b5f9f53bbae69db2dd5661))

* Add option to train MDAS, enable RAVEL with custom SAEs ([`e030f12`](https://github.com/adamkarvonen/SAEBench/commit/e030f126a5a70642f9ac55129833783d15a4c331))

* Implement ravel with generations ([`e2f7bc5`](https://github.com/adamkarvonen/SAEBench/commit/e2f7bc5fcac6838a21aabadc81be35516bc78a34))

* Start running with new RAVEL dataset ([`12e658f`](https://github.com/adamkarvonen/SAEBench/commit/12e658fa5cf442922a14e6d255b219c1526ae21a))

* RAVEL dataset refactor ([`1e21377`](https://github.com/adamkarvonen/SAEBench/commit/1e21377a9de22bf20be5fd1ea7ad05b29e25f1b8))

* ravel DAS implementation ([`c66a604`](https://github.com/adamkarvonen/SAEBench/commit/c66a6042827bb63618e4c51e1d257ae5c4803326))

* Remove recursive prompt selection due to random stack overflow issues ([`1481e30`](https://github.com/adamkarvonen/SAEBench/commit/1481e30f51f28d182a217832868f99749a87f5af))

* Enable running eval on SAE Lens SAEs ([`b83309d`](https://github.com/adamkarvonen/SAEBench/commit/b83309d64604f9f03cba1afdbaa848b6aade97ea))

* Initial run of RAVEL in SAE Bench ([`ceb23c6`](https://github.com/adamkarvonen/SAEBench/commit/ceb23c69e8e679cd256aca1a5c767d243d47de7f))

* Clean up ([`70d567e`](https://github.com/adamkarvonen/SAEBench/commit/70d567e4d0c86e4365dc22461ac3ceeb54e865bf))

* Compute cause / iso scores for validation data, test a supervised baseline ([`4b5b734`](https://github.com/adamkarvonen/SAEBench/commit/4b5b734919f3c4f0a482ab6d35cbb0cb63c05238))

* Add isolation prompts ([`b7cbb56`](https://github.com/adamkarvonen/SAEBench/commit/b7cbb56e7b948daa5773307cf070960eb36f4558))

* DBM working with debug information ([`0f4815b`](https://github.com/adamkarvonen/SAEBench/commit/0f4815b24b24ef01bb4ec97301e5d14e03bb0935))

* Tested working DBM intervention ([`bbc1d0c`](https://github.com/adamkarvonen/SAEBench/commit/bbc1d0cc563ec7861b6146e6a342e39f50b5684c))

* Initial stab at MDBM ([`5516969`](https://github.com/adamkarvonen/SAEBench/commit/55169693e5ab58a80f743bfdd93707b425fbaf5c))

* Temp nnsight commit ([`10b86a7`](https://github.com/adamkarvonen/SAEBench/commit/10b86a73c801f318877c92d387669200c8b6c96a))

* Merge pull request #61 from Stefan-Heimersheim/fix/correct_explained_variance

Fix explained variance calculation ([`52fe993`](https://github.com/adamkarvonen/SAEBench/commit/52fe993418100c457ae4fba2d92dfa79bc64eb64))

* Batched version of the variance computation ([`3b58c67`](https://github.com/adamkarvonen/SAEBench/commit/3b58c672e74678642859d9cc1baff004b4fce1d4))

* Use conventional explained variance definition ([`7fb3ed3`](https://github.com/adamkarvonen/SAEBench/commit/7fb3ed3556e77d56bc81d0e314c769f1b75bc9f7))

* Comparing explained variance definitions ([`baadb26`](https://github.com/adamkarvonen/SAEBench/commit/baadb26618c8beb50770aa47695159c7784fc5b0))

## v0.4.0 (2025-02-22)

### Chore

* chore: making test less flaky ([`3effa18`](https://github.com/adamkarvonen/SAEBench/commit/3effa183b472f6c589850b7382c8d30c6ef39850))

* chore: fix updated torch types ([`4c46da6`](https://github.com/adamkarvonen/SAEBench/commit/4c46da6610b3bba7df2910cd0397a9926c4a9264))

* chore: fixing linting errors and adding precommit hook ([`85f6241`](https://github.com/adamkarvonen/SAEBench/commit/85f624182319c36f9b3d9bd520e95e61a1264bd1))

### Feature

* feat: allow setting the artifacts path ([`2a4b4dc`](https://github.com/adamkarvonen/SAEBench/commit/2a4b4dc1bc9bbe401b46c36097bd647743c4af80))

### Fix

* fix: gracefully handle slashes in model filename for autointerp ([`5d6464a`](https://github.com/adamkarvonen/SAEBench/commit/5d6464a8ee838ab036d5051ab0304a92ba890baa))

* fix: fix typing and updating mdl for saelens &gt;=5.4.0 ([`802d1c3`](https://github.com/adamkarvonen/SAEBench/commit/802d1c34019b505227a460ae71febc5424ce36c1))

* fix: load probe class with weights_only = False ([`f05bf40`](https://github.com/adamkarvonen/SAEBench/commit/f05bf408ebbec97f2895af89facb7c1fbf90a72d))

* fix: Update README to include eval output schema update instructions ([`f0adee2`](https://github.com/adamkarvonen/SAEBench/commit/f0adee26f83731cc11598d82c1be7abdc2b0fcfd))

* fix: Update json schema jsons ([`2b2a6d3`](https://github.com/adamkarvonen/SAEBench/commit/2b2a6d36b6a1a1f561cc86ecdd0f425568bd2378))

### Unknown

* Merge pull request #60 from chanind/deflaking-test

chore: making test less flaky ([`963f2e8`](https://github.com/adamkarvonen/SAEBench/commit/963f2e8e36dfc2b0574995a875e936801d4028b2))

* Remove threshold from state dict if we aren&#39;t using it ([`d91a218`](https://github.com/adamkarvonen/SAEBench/commit/d91a218b4cc4ac6c164d0e1b739c8437901c7acd))

* Merge pull request #59 from chanind/artifacts-path-option

feat: allow setting the artifacts path ([`53901a2`](https://github.com/adamkarvonen/SAEBench/commit/53901a215035840a642fd2b29dc619140757e27f))

* adding artifacts_path to unlearning eval ([`ce1de32`](https://github.com/adamkarvonen/SAEBench/commit/ce1de32470187be0228e8fc83a5098c61048b0be))

* Merge pull request #58 from chanind/fixing-types

chore: fix updated torch types ([`849018f`](https://github.com/adamkarvonen/SAEBench/commit/849018fb584946762b201cbbf155b87132880325))

* Merge pull request #57 from chanind/fix-slash-in-model-name-autointerp

fix: gracefully handle slashes in model filename for autointerp ([`11b2e38`](https://github.com/adamkarvonen/SAEBench/commit/11b2e3851c9212996a2d9994f0315e33d4cc53d5))

* By default we don&#39;t use a threshold for custom topk SAEs ([`60579ed`](https://github.com/adamkarvonen/SAEBench/commit/60579ed19a6281956621283bada7be2e76a7b583))

* Merge pull request #56 from chanind/type-fixes

fix: fix typing and updating mdl for saelens &gt;=5.4.0 ([`0888d07`](https://github.com/adamkarvonen/SAEBench/commit/0888d07876caff50d6e292d11950cee349fb0904))

* Merge pull request #55 from chanind/precommit-check

chore: fixing linting errors and adding precommit hook ([`7ac7ced`](https://github.com/adamkarvonen/SAEBench/commit/7ac7ced7bca2d47247f0c7fa5e4ad7f6437dec60))

* Fix SAE Bench SAEs repo names ([`18dc457`](https://github.com/adamkarvonen/SAEBench/commit/18dc4571ba027f09c8d487658a0b615977c1a127))

* Prevent potential division by zero ([`92315dd`](https://github.com/adamkarvonen/SAEBench/commit/92315dd4755c472bb6dfd8548a69f769d8644c6c))

* Add optional pinned dependencies ([`e74f0cf`](https://github.com/adamkarvonen/SAEBench/commit/e74f0cfac9943ae992d62fc3974cb457cb273df2))

* Calculate featurewise statistics in demo ([`5204b48`](https://github.com/adamkarvonen/SAEBench/commit/5204b4822c66a838d9c9221640308e7c23eda00a))

* Improve documentation on custom SAE usage ([`f15fe53`](https://github.com/adamkarvonen/SAEBench/commit/f15fe53281238c31c72f6646a53b9650e7b0a5f4))

* Merge pull request #53 from adamkarvonen/hide_absorption_stddev

hide stddev from default display for absorption ([`155afbc`](https://github.com/adamkarvonen/SAEBench/commit/155afbca50a9ffe6cf72c81796997e6daa475658))

* hide stddev from default display for absorption ([`d970f05`](https://github.com/adamkarvonen/SAEBench/commit/d970f054ca9918ea6ecd6a228106377d512eb55c))

* Merge pull request #52 from adamkarvonen/update_scr_tpp

update scr_tpp_schema to show top 20 by default ([`f551e7b`](https://github.com/adamkarvonen/SAEBench/commit/f551e7bf6d8c615b50e524c3e07091a8e3c73fa0))

* update scr_tpp_schema to show top 20 by default ([`59320e2`](https://github.com/adamkarvonen/SAEBench/commit/59320e2d6bc44b067b3a8476b32535ce7238f68e))

* Merge pull request #51 from adamkarvonen/update_schema_jsons

fix: Update eval output schema jsons ([`7b2021c`](https://github.com/adamkarvonen/SAEBench/commit/7b2021cbf57d8b491920f168caf907ed05a43c49))

* Add computational requirements ([`9b621a9`](https://github.com/adamkarvonen/SAEBench/commit/9b621a93a865c3b16262dc129689b5f824840308))

* Improve graphing notebook, include matryoshka results in graphs ([`f2d1d98`](https://github.com/adamkarvonen/SAEBench/commit/f2d1d982515d2dee706eb23a1ca459b308988764))

* Merge pull request #50 from chanind/lint-and-type-check

chore: Adding formatting, linting and type checking ([`a0fb5e9`](https://github.com/adamkarvonen/SAEBench/commit/a0fb5e90a82a0414ca9be0511ec3df44af433f2f))

* adding README and Makefile with helpers ([`7452eca`](https://github.com/adamkarvonen/SAEBench/commit/7452eca1a13563bb5a625409aad6b46b319a4076))

* fixing linting and type-checking issues ([`e663e3a`](https://github.com/adamkarvonen/SAEBench/commit/e663e3a097fc1c49b51096b2455c5639aaa099f8))

* formatting with ruff ([`14dad45`](https://github.com/adamkarvonen/SAEBench/commit/14dad450954fad4dd55e8480204d227723bf4efe))

* Check that unlearning data exists before running unlearning eval ([`294b25c`](https://github.com/adamkarvonen/SAEBench/commit/294b25ca1f531f2fd97333292dd30315a009e225))

* Improve export notebook ([`e2b0b3c`](https://github.com/adamkarvonen/SAEBench/commit/e2b0b3c57a3d256998f8bda15cdb21542f226d1a))

* Improve graphing utils ([`661920d`](https://github.com/adamkarvonen/SAEBench/commit/661920d17c52ca1c479c21bbf5c68b40c9d50222))

* Fix spelling ([`8c0df93`](https://github.com/adamkarvonen/SAEBench/commit/8c0df931c606bbfd83d0688442fed2a6d23c37f6))

* Add standard deviation for absorption / autointerp, store results per class for sparse probing / tpp for potential error bars ([`141aff7`](https://github.com/adamkarvonen/SAEBench/commit/141aff72928f7588c1451bed47c401e1d565d471))

* Use GPU probing in correct location ([`ec5efa8`](https://github.com/adamkarvonen/SAEBench/commit/ec5efa820ceb6e88d53667f247bb2a09efca609f))

## v0.3.2 (2025-01-14)

### Fix

* fix: use GPU for llm probing ([`ba0956e`](https://github.com/adamkarvonen/SAEBench/commit/ba0956e74ee541716823b5df98d91576b6af05a1))

### Unknown

* Don&#39;t hardcode the device for unlearning ([`a594ee6`](https://github.com/adamkarvonen/SAEBench/commit/a594ee6e8bc751f0f1502e5db5410992d6289793))

* Update unlearning data path ([`443761d`](https://github.com/adamkarvonen/SAEBench/commit/443761dcac34aa50201f2edc2564c622ff861eaa))

## v0.3.1 (2025-01-14)

### Fix

* fix: pass device into core evals ([`e6651ea`](https://github.com/adamkarvonen/SAEBench/commit/e6651ea60b87973cda31f3801dc2f34e196052bf))

### Unknown

* fold W_dec norm when loading SAE Lens SAEs ([`511d51a`](https://github.com/adamkarvonen/SAEBench/commit/511d51a5ef58997796a4769dff5a13f21f5e934b))

* Change default sparse probing k values ([`271a9d4`](https://github.com/adamkarvonen/SAEBench/commit/271a9d4a61b35d4f4d11570ae01a70ec5c2b649d))

## v0.3.0 (2025-01-13)

### Feature

* feat: Add a frac alive calculation to core ([`0399550`](https://github.com/adamkarvonen/SAEBench/commit/039955013df919bd3d5cadc9a03bc96810351f37))

### Unknown

* added absorption fraction metric (#48)

feat: added absorption fraction metric

* Small fixes

* remove unused FeatureAbsorptionCalculator._filter_prompts function

---------

Co-authored-by: Demian Till &lt;demian.till@cambridgeconsultants.com&gt; ([`7545ee3`](https://github.com/adamkarvonen/SAEBench/commit/7545ee345a1f2290625b4dd38e12f75084190ffd))

* Add a script for organizing and uploading results ([`4689129`](https://github.com/adamkarvonen/SAEBench/commit/468912987d952d86e4a4eb952dbada577331dc7d))

* Calculate featurewise statistics by default ([`bca84ca`](https://github.com/adamkarvonen/SAEBench/commit/bca84cabc8cd60f8b15f37668faece7bbd9adc23))

## v0.2.0 (2025-01-09)

### Feature

* feat: add misc core metrics ([`2c731f6`](https://github.com/adamkarvonen/SAEBench/commit/2c731f611ed1090ea4c04857736cc5e3b9c2af0f))

### Unknown

* Make sure grad is enabled for absorption tests ([`bd25ca0`](https://github.com/adamkarvonen/SAEBench/commit/bd25ca01c92b8fc231bbc5a12f509720e00efd43))

## v0.1.0 (2025-01-09)

### Feature

* feat: EvalOutput and EvalConfig base classes to allow easy JSON schema export ([`537219a`](https://github.com/adamkarvonen/SAEBench/commit/537219a8c4306b21a1815731fac5f3effdf450d8))

### Fix

* fix: eval_result_unstructured should be optional ([`38e81b0`](https://github.com/adamkarvonen/SAEBench/commit/38e81b0acd73e45c398ceea17fec3aac79228ebf))

* fix: dump to json file correctly ([`5f1cf15`](https://github.com/adamkarvonen/SAEBench/commit/5f1cf15d3f5edfe126be2d93d8a52c9e7a585755))

### Unknown

* git commit -m &#34;fix: add missing __init__.py&#34; ([`20b20f2`](https://github.com/adamkarvonen/SAEBench/commit/20b20f2a2d42ad2d01235c73ad4942e328800c5b))

* Merge pull request #47 from chanind/packaging

feat: Setting up Python packaging and autodeploy with Semantic Release ([`e52a418`](https://github.com/adamkarvonen/SAEBench/commit/e52a418276a88701f4f317bda63b67b4bbd44031))

* Merge branch &#39;main&#39; into packaging ([`9bc22a4`](https://github.com/adamkarvonen/SAEBench/commit/9bc22a402d7691a3a6f809be174c7fdf90d003e6))

* Update SAE Bench demo to use new graphing functions ([`9bbfdc5`](https://github.com/adamkarvonen/SAEBench/commit/9bbfdc5d4bb4836a9f4bc58a99b7b6d00a12214d))

* Merge branch &#39;main&#39; into packaging ([`bb10234`](https://github.com/adamkarvonen/SAEBench/commit/bb10234445d096c4be364cf0183733a86693d9b5))

* Add option to pass in arbitrary sae_class ([`e450661`](https://github.com/adamkarvonen/SAEBench/commit/e450661e430979f8437d59d2324d1eee72a4f4fd))

* Mention dictionary_learning ([`c140e71`](https://github.com/adamkarvonen/SAEBench/commit/c140e7180c81c88f57c2d1114413bd9bfb67153e))

* Update graphing notebook to work with filenames ([`dc6f951`](https://github.com/adamkarvonen/SAEBench/commit/dc6f9513038c5224918e5365d7680190ccca6fa2))

* deprecate graphing notebook ([`67118ee`](https://github.com/adamkarvonen/SAEBench/commit/67118ee6785b1830782ddf994a4349276e1bf080))

* Use a smaller batch size for unlearning ([`3a099d2`](https://github.com/adamkarvonen/SAEBench/commit/3a099d28607f73d3b816ded300591b86ae42d1b6))

* switching to poetry and setting up CI ([`a9af271`](https://github.com/adamkarvonen/SAEBench/commit/a9af2713f1a90c4b15ab6ba577b1f2242f88e21b))

* migrating to sae_bench base dir ([`bb8e145`](https://github.com/adamkarvonen/SAEBench/commit/bb8e145dec79af603dbb0a56d37f5f63153cff25))

* Reduce memory usage by only caching required activations ([`f026998`](https://github.com/adamkarvonen/SAEBench/commit/f026998294e0e7e231aea599bdfde0a278f1d08c))

* Remove debugging check ([`8ea7162`](https://github.com/adamkarvonen/SAEBench/commit/8ea7162fd37f55eac298c6e6a23970252db7bc9e))

* Add sanity checks before major run ([`0908b18`](https://github.com/adamkarvonen/SAEBench/commit/0908b187dd7be2d37d80311405126b72d66c0f84))

* Improve normalization check ([`16a3c0e`](https://github.com/adamkarvonen/SAEBench/commit/16a3c0e2e95462717d8dd0b3c07dbd6e98911110))

* Add normalization for batchtopk SAEs ([`6a031bd`](https://github.com/adamkarvonen/SAEBench/commit/6a031bd4b25004a74c8f38e584252702fc0f68d5))

* Add matroyshka loader ([`1078899`](https://github.com/adamkarvonen/SAEBench/commit/107889901c813248a51c5e2f4b87aee7a90c3a5c))

* Add pythia 160m ([`b219497`](https://github.com/adamkarvonen/SAEBench/commit/b2194975f9522f5f45b073b8a008d90b675574a7))

* simplify process of evaluating dictionary learning SAEs ([`c2dca52`](https://github.com/adamkarvonen/SAEBench/commit/c2dca529b5625ca102521fd4e5f35c985913fca9))

* Add a script to run evals on dictionary learning SAEs ([`3f4139b`](https://github.com/adamkarvonen/SAEBench/commit/3f4139be2c5932eab9d92a78e316c20261d5def5))

* Make the layer argument optional ([`e53675d`](https://github.com/adamkarvonen/SAEBench/commit/e53675d73a733577d9dab25bcecf567d72a54022))

* Add batch_top_k, top_k, gated, and jump_relu implementations ([`9a7fce8`](https://github.com/adamkarvonen/SAEBench/commit/9a7fce8baebb7c28382681e88b8d319b6630c051))

* Add a function to test the saes ([`864b4b3`](https://github.com/adamkarvonen/SAEBench/commit/864b4b355ac13fa7afc7191608091b8340b19261))

* Update demo for new relu sae setup ([`5d04ce5`](https://github.com/adamkarvonen/SAEBench/commit/5d04ce5d661712eec7982c23d30b124aad61d9ef))

* Ensure loaded SAEs are on correct dtype and device ([`a5d6d62`](https://github.com/adamkarvonen/SAEBench/commit/a5d6d620f91346fba278e8ba36b547343e7d1856))

* Create a base SAE class ([`8fcc9fe`](https://github.com/adamkarvonen/SAEBench/commit/8fcc9fe5497fabbdf34eb20fa703e3400b626045))

* Add blog post link ([`2d47229`](https://github.com/adamkarvonen/SAEBench/commit/2d47229d48a992f485bd2b8ec6a4bbbb68966dc7))

* cleanup README ([`0e724df`](https://github.com/adamkarvonen/SAEBench/commit/0e724dfd18c2a5751c2c26ab0186b80badd03890))

* Clean up graphing notebook ([`c08f3f5`](https://github.com/adamkarvonen/SAEBench/commit/c08f3f5cb701a15feb1f7781f17d20bf050a5823))

* Graph results for all evals in demo notebook ([`29ac97b`](https://github.com/adamkarvonen/SAEBench/commit/29ac97bfeb106a287bb0decb003079990ab0c72e))

* Clean up for release ([`1c9822c`](https://github.com/adamkarvonen/SAEBench/commit/1c9822c7270aa7f9a6a61379603a981e837096b0))

* Include baseline pca in every graph. ([`a45afd2`](https://github.com/adamkarvonen/SAEBench/commit/a45afd26ff427fed11d2f1c13f88b43528f222be))

* Clean up plot legends, support graphing subplots ([`7ade8b0`](https://github.com/adamkarvonen/SAEBench/commit/7ade8b06a9102288fc3e8168bed70b5b9de344c7))

* Merge pull request #45 from adamkarvonen/update_jsonschemas

update jsonschemas ([`879c7ca`](https://github.com/adamkarvonen/SAEBench/commit/879c7ca631e2aefda703333480e42d9b0fc1468b))

* update jsonschemas ([`a14d465`](https://github.com/adamkarvonen/SAEBench/commit/a14d4657987a39e68772ae62391f5027fdb60514))

* Use notebook as default demo, mention in README ([`298796b`](https://github.com/adamkarvonen/SAEBench/commit/298796bcd0f516dc431a20a6060eef7ac0b67101))

* Minor fixes to demo ([`05808c7`](https://github.com/adamkarvonen/SAEBench/commit/05808c7c6c8f50ba138fe1c3c1fa9cab394e9d2f))

* Add missing batch size argument ([`877f2e7`](https://github.com/adamkarvonen/SAEBench/commit/877f2e7a9fb0ab7d83b8ee748ad2fba31671e35d))

* Fixes for changes to eval config formats ([`e0cb629`](https://github.com/adamkarvonen/SAEBench/commit/e0cb6297d556ac1fd3b4af3d173bbd9e8842f6be))

* Add an optional best of k graphing cell ([`081b59c`](https://github.com/adamkarvonen/SAEBench/commit/081b59cce51d4703eb49cb5638b95b0ecce10d46))

* Ignore any folder containing &#34;eval_results&#34; ([`12f8d66`](https://github.com/adamkarvonen/SAEBench/commit/12f8d665a3240576f6d03760a455d430a9edf584))

* Add cell to add training tokens to config dictionaries ([`38173c9`](https://github.com/adamkarvonen/SAEBench/commit/38173c992279986e527025e2dbd61e63d071f49a))

* Also plot all sae bench checkpoints ([`93563e0`](https://github.com/adamkarvonen/SAEBench/commit/93563e00ea230630d810d6eb3e0288ed5fae61b0))

* Add eval links ([`2216f99`](https://github.com/adamkarvonen/SAEBench/commit/2216f99365c8a0510d3d4a2e286aa6f1133c0edc))

* rename core results to match convention ([`51e47fd`](https://github.com/adamkarvonen/SAEBench/commit/51e47fd3e55e3a95bbbbcd37cbf4485e1c7b1fa2))

* Ignore autointerp with generations when downloading ([`aa20644`](https://github.com/adamkarvonen/SAEBench/commit/aa20644d3565d7e0ac564566586846f8dce6915c))

* Use != instead of &gt; for L0 measurement ([`83504b7`](https://github.com/adamkarvonen/SAEBench/commit/83504b7032e162466fd4d64cd820a7fc39550739))

* Add utility cell for removing llm generations ([`67c9b03`](https://github.com/adamkarvonen/SAEBench/commit/67c9b0381e8bbf0ef5a845dee64aa2b955a47213))

* Add utility cell for splitting up files by release name ([`3cc51ea`](https://github.com/adamkarvonen/SAEBench/commit/3cc51ea69389ae4249c46c27dcd848310ea3e758))

* Add force rerun option to core, match sae loading to other evals ([`8676d5d`](https://github.com/adamkarvonen/SAEBench/commit/8676d5d6a85b6524c74e776e2d7e2554ca0d324a))

* Improve plotting of results ([`89e5567`](https://github.com/adamkarvonen/SAEBench/commit/89e55671b1681ab99aba5187daf5b0a51c36c0a1))

* Consolidate SAE loading and output locations ([`293b385`](https://github.com/adamkarvonen/SAEBench/commit/293b3851ff78c7d9f4ea675d18085b13ac87725a))

* Plot generator for SAE Bench ([`c2cb78e`](https://github.com/adamkarvonen/SAEBench/commit/c2cb78e52d15054cc5bdc5c8b5c7b4ba6c94de8c))

* Add utility notebook for adding sae configs ([`8508a01`](https://github.com/adamkarvonen/SAEBench/commit/8508a0154376b6d04177fbfa2d52b024fba6c7ca))

* Improve custom SAE usage ([`e959f65`](https://github.com/adamkarvonen/SAEBench/commit/e959f65b76d42f940fd65603507c68002c5280f0))

* Improve graphing ([`490cd2a`](https://github.com/adamkarvonen/SAEBench/commit/490cd2ab33f811b4ccf3b8ef1757f3a8dc8573b5))

* Fix failing tests ([`ed88f65`](https://github.com/adamkarvonen/SAEBench/commit/ed88f6549b27c93034f64044166b95a5b4b804fd))

* match core output filename with others ([`8ca0787`](https://github.com/adamkarvonen/SAEBench/commit/8ca0787cff38d57fbd3d81546429ac74fc1e3dc5))

* Remove del sae flag ([`feaf1f8`](https://github.com/adamkarvonen/SAEBench/commit/feaf1f8f6b76eac2bded01dac9181b3a55b85bf2))

* Add current status to repo ([`9c95af7`](https://github.com/adamkarvonen/SAEBench/commit/9c95af7faf762dc3f0ece3b4367efa845c5c677a))

* Add sae config to output file ([`b2fbd6d`](https://github.com/adamkarvonen/SAEBench/commit/b2fbd6d292b651a631ad74e1bff5c2b0f5cd5fb6))

* Add a flag for k sparse probing batch size ([`6f2e38f`](https://github.com/adamkarvonen/SAEBench/commit/6f2e38f6481933249b70185f9d3b68737eac44a1))

* Merge pull request #44 from adamkarvonen/absorption-tweaks-2

improving memory usage of k-sparse probing ([`6ae8235`](https://github.com/adamkarvonen/SAEBench/commit/6ae8235696eab99c1bf81e7ef1413ceafcb29699))

* improving memory usage of k-sparse probing ([`841842a`](https://github.com/adamkarvonen/SAEBench/commit/841842aaccbb8a086e3d090cffa58ab1d13db2c9))

* Merge pull request #43 from adamkarvonen/fake_branch

single line update ([`7984d50`](https://github.com/adamkarvonen/SAEBench/commit/7984d508b0023b73061961d091f28a97928aaa8f))

* single line update ([`d9637e1`](https://github.com/adamkarvonen/SAEBench/commit/d9637e12306d0a7e9c0a0d1a732712080360ec90))

* Add documentation to demo notebook ([`2e170e1`](https://github.com/adamkarvonen/SAEBench/commit/2e170e1f4d5bcb8e531518e32289bfac771b32d9))

* adapted graphing to np result filestructure ([`3629b90`](https://github.com/adamkarvonen/SAEBench/commit/3629b9067f2518700d317ca31a3d1cdfe077576e))

* Improve reduced memory script ([`ecb9f46`](https://github.com/adamkarvonen/SAEBench/commit/ecb9f46699a1576abcfe36d2b4730fb9dd0e24c5))

* Script for evaluating 1M width SAEs ([`63a6783`](https://github.com/adamkarvonen/SAEBench/commit/63a67834650f1ae9728a1819b1244a13b0d29963))

* Use expandable segments to reduce memory usage ([`4f3967d`](https://github.com/adamkarvonen/SAEBench/commit/4f3967dba8c69edffcaecf76d905652074a67a3b))

* Delete SAE at the correct location in the for loop ([`ff0beda`](https://github.com/adamkarvonen/SAEBench/commit/ff0beda5911d563f6a62da07fda7aa52c1705148))

* Shell script for running 65k width SAEs on 24 GB GPUs ([`9b0bd9d`](https://github.com/adamkarvonen/SAEBench/commit/9b0bd9dd407cbe576a2fd7f57681fe1532d4f5da))

* Delete sae at end of loop to lower memory usage. Primarily required for 1M width SAEs ([`08f9755`](https://github.com/adamkarvonen/SAEBench/commit/08f9755ab7ccbeab0af0cbf6a4403375c7c89158))

* Add absorption ([`b2e89c9`](https://github.com/adamkarvonen/SAEBench/commit/b2e89c91521897d8fef75d9842430392093d6eac))

* Add note on usage ([`07cbf3c`](https://github.com/adamkarvonen/SAEBench/commit/07cbf3c441ab41b36660e9cf930f9521f7049b4d))

* Add shell scripts for running all evals ([`a832e09`](https://github.com/adamkarvonen/SAEBench/commit/a832e09c1f5950ed58655cb8c4a222a3a69d46f0))

* add 9b-it unlearning precomputed artifacts ([`93502c0`](https://github.com/adamkarvonen/SAEBench/commit/93502c0065b910051985d0f5ab0df7179e70a5cb))

* Add example of running all evals to notebook ([`473081d`](https://github.com/adamkarvonen/SAEBench/commit/473081d649fd5b84672138e5f3cc80dfaa770c60))

* Clean up filename ([`a067c5c`](https://github.com/adamkarvonen/SAEBench/commit/a067c5cc5c67c86c69fa8ee148dc15f9df23232e))

* Create a demo of using custom SAEs on SAE bench ([`49d5ecd`](https://github.com/adamkarvonen/SAEBench/commit/49d5ecd8721356087e3fbf89aaa3f9ebb1d473f0))

* Move warnings to main function, raise error if not instruct tuned ([`e798adf`](https://github.com/adamkarvonen/SAEBench/commit/e798adf387996ef179913f4449ec9fdc8ceb5e7b))

* perform calculations with a running sum to avoid underflow ([`d842a1f`](https://github.com/adamkarvonen/SAEBench/commit/d842a1fe42edc9d0c8b972f920eb14d14261f54f))

* Do probe attribution calculation in original dtype for memory savings ([`366dc4c`](https://github.com/adamkarvonen/SAEBench/commit/366dc4cf632cbda1ea60c5b3a4d5ca0dc8443390))

* Use api key file instead of command line argument ([`bb48a6c`](https://github.com/adamkarvonen/SAEBench/commit/bb48a6cdf2503938c8ba447a99f0469a54321b66))

* Add flags to reduce VRAM usage ([`322334a`](https://github.com/adamkarvonen/SAEBench/commit/322334a3f6e2605b50dbfddae8ae4de73e57046c))

* fix unlearning test ([`5039e5e`](https://github.com/adamkarvonen/SAEBench/commit/5039e5e809a8aef3c74ede3cbc2d9ab01672edee))

* add optional flag to reduce peak memory usage ([`735f988`](https://github.com/adamkarvonen/SAEBench/commit/735f98811c7fef9746aed094f7c657a61915e11e))

* Ignore core model name flag for now ([`43ef711`](https://github.com/adamkarvonen/SAEBench/commit/43ef711bdc819203a93037dfc3c9fa317483c12e))

* Don&#39;t try set random seed if it&#39;s none ([`d1d6f72`](https://github.com/adamkarvonen/SAEBench/commit/d1d6f7224ca0deda9d2362d4e54b3cdbc33fd16d))

* Make eval configs consistent, require model names in all eval arguments. ([`d37e77c`](https://github.com/adamkarvonen/SAEBench/commit/d37e77c2fbe6972083a3d5670e44dbe9bd58c20d))

* Add ability to pass in random seed and sae / llm batch size ([`d8f026b`](https://github.com/adamkarvonen/SAEBench/commit/d8f026bb33422b5660155dc9e6b97557ae970299))

* Describe how values are set within eval configs ([`365fb40`](https://github.com/adamkarvonen/SAEBench/commit/365fb40aafcf7bda6477acbd17d9d5a7061db571))

* Always ignore the bio forget corpus ([`3e6d36f`](https://github.com/adamkarvonen/SAEBench/commit/3e6d36f9eb605185fe3214b6045c539b1df8d098))

* Use util function to convert str to dtype ([`7281627`](https://github.com/adamkarvonen/SAEBench/commit/7281627d5e479cf972ebca5764f6ee216227f58f))

* update graphing scripts ([`ff38240`](https://github.com/adamkarvonen/SAEBench/commit/ff382404ee4b34cba20a09174fa1f8cb9e80abfc))

* Merge pull request #39 from adamkarvonen/add_9b

add gemma-2-9b default DTYPE and BATCH_SIZE ([`164b6f5`](https://github.com/adamkarvonen/SAEBench/commit/164b6f53b0814610799175680f1523d34cf49238))

* also add for 9b-it ([`b93f3c9`](https://github.com/adamkarvonen/SAEBench/commit/b93f3c9ef7b2011b78217d99810017a90ba6f945))

* add gemma-2-9b ([`8030c03`](https://github.com/adamkarvonen/SAEBench/commit/8030c0394ce123d86c6b4af764143d067d8598f9))

* Update regexes and test data to match new SAE Bench SAEs ([`6da4692`](https://github.com/adamkarvonen/SAEBench/commit/6da46928230bf3003981f285262c0f5b51fe3abb))

* Update outdated reference, don&#39;t get api_key if not required ([`da9a2dc`](https://github.com/adamkarvonen/SAEBench/commit/da9a2dc7bd29520761ed3a196f3912a42c010ef9))

* Add ability to pass in flag for computing featurewise statistics, default it to false ([`f6430af`](https://github.com/adamkarvonen/SAEBench/commit/f6430afc9152bc7d8ab894fc1e047fc07e6c6427))

* Move str_to_dtype() to general utils ([`8ab32f9`](https://github.com/adamkarvonen/SAEBench/commit/8ab32f9464710131bd3e3a16077eefecbfbb8b23))

* Pass in a string dtype ([`f49d41c`](https://github.com/adamkarvonen/SAEBench/commit/f49d41c3b2e1f59ab18965d607c38a6e1c76fb86))

* Merge pull request #35 from adamkarvonen/add_pca

Add pca ([`f4fbd0c`](https://github.com/adamkarvonen/SAEBench/commit/f4fbd0cb09ccff243b0f0d2eb561b11842afa8e3))

* Delete old sae bench data ([`55f9b6f`](https://github.com/adamkarvonen/SAEBench/commit/55f9b6f4d36ec6cb1dab2cf7f768ce3f7ada700f))

* Mention disk space, fix repo name ([`04a2b01`](https://github.com/adamkarvonen/SAEBench/commit/04a2b01ce7194d4d393feefcc33720731b998822))

* mention WMDP access ([`26816e5`](https://github.com/adamkarvonen/SAEBench/commit/26816e5c0c8167aba9ab06d368e92e2f3895bfe8))

* Be consistent when setting default dtype ([`3ed82b3`](https://github.com/adamkarvonen/SAEBench/commit/3ed82b3f2b1aa93439cf309f4219a557736e98b1))

* Rename baselines to custom_saes ([`067bb79`](https://github.com/adamkarvonen/SAEBench/commit/067bb79b1de7f286cbc4078b23a6e4a1e39dd4b0))

* Rename shift to SCR ([`bbbdfdc`](https://github.com/adamkarvonen/SAEBench/commit/bbbdfdc5dd2da6bb969b832aad26fb01a94f3fac))

* correctly save and load state dict ([`35d64c8`](https://github.com/adamkarvonen/SAEBench/commit/35d64c81d6ff113572f2705fd18196ef274be633))

* Just use the global PCA mean ([`2317de9`](https://github.com/adamkarvonen/SAEBench/commit/2317de94285199c0231cde79fc4b9d982fa4f065))

* Increase test tolerance, remove cli validation as other evals aren&#39;t using it ([`395095b`](https://github.com/adamkarvonen/SAEBench/commit/395095b30cbe7ae11d9a8ef36fdd62e2a6115240))

* Match core eval config to others with dtype usage ([`9197e3f`](https://github.com/adamkarvonen/SAEBench/commit/9197e3f8dd5f10dd2a471dcd098d8ccad9fddc64))

* Also check for b_mag so we don&#39;t ignore gated SAEs bias ([`edd0de2`](https://github.com/adamkarvonen/SAEBench/commit/edd0de209f2dc7deef039281b672037d906ea452))

* consolidate save locations of artifacts and eval results ([`c9a18b0`](https://github.com/adamkarvonen/SAEBench/commit/c9a18b0c5014d74b060ae178a6a6ea525dd0f98b))

* revert eval_id change for now ([`2cfbcc0`](https://github.com/adamkarvonen/SAEBench/commit/2cfbcc02c4a24b9b790156cfef8af048bdd913cb))

* Save fake encoder bias as a tensor of zeros ([`4ccbec6`](https://github.com/adamkarvonen/SAEBench/commit/4ccbec6079d6c9f842f9bf7c918882b1749a558d))

* Ensure that sae_name is unique ([`cce38b6`](https://github.com/adamkarvonen/SAEBench/commit/cce38b66bf479457d6bff532cb59fb19ece2ca30))

* Change default results location ([`6987235`](https://github.com/adamkarvonen/SAEBench/commit/6987235fe6c34f218ae95079dc2caabc096d0ff7))

* Also compare residual stream as a baseline ([`d528f3b`](https://github.com/adamkarvonen/SAEBench/commit/d528f3bb026387d3fabc382d3a29e5144ab045e5))

* Don&#39;t require sae to have a b_enc ([`c979427`](https://github.com/adamkarvonen/SAEBench/commit/c979427b7f9e2d9895b180de1f2e2433758aae3f))

* Include model name in tokens filename ([`a255df0`](https://github.com/adamkarvonen/SAEBench/commit/a255df0122d7707e138f2d20390386e9175bc9bb))

* Check if file exists ([`f8c9ab2`](https://github.com/adamkarvonen/SAEBench/commit/f8c9ab29c3794e538903c925c1ae0e8ff3f6a544))

* Fix regex usage demo ([`1c4117d`](https://github.com/adamkarvonen/SAEBench/commit/1c4117d20806b46f3c21848523dd7d91c38ca0ba))

* remove outdated import ([`9dddb3e`](https://github.com/adamkarvonen/SAEBench/commit/9dddb3edc8f01521f649d14f1aad9e565b526129))

* Simplify custom SAE usage demonstration ([`542c659`](https://github.com/adamkarvonen/SAEBench/commit/542c659bef1ca526e48eabe9c4b484bc67a7dc21))

* Benchmark autointerp scores on mlp neurons and saes ([`4ba6cba`](https://github.com/adamkarvonen/SAEBench/commit/4ba6cbaa2fdfe72fb4e2af2fab6a6dceb0525b23))

* Simplify code by storing dtype as a string ([`963c9c2`](https://github.com/adamkarvonen/SAEBench/commit/963c9c237985e06ea532d463c89cad73c5e3fb86))

* Add option to set dtype for core/main.py ([`d649826`](https://github.com/adamkarvonen/SAEBench/commit/d64982688c4c57149f36bc3b52fb8a538d115edb))

* Pass in the optional flag to save activations ([`d0b8091`](https://github.com/adamkarvonen/SAEBench/commit/d0b8091e6ff0cb12850ec6d43fd1d6d112bb7b81))

* Don&#39;t check for is SAE to enable use with custom SAEs ([`36cfba8`](https://github.com/adamkarvonen/SAEBench/commit/36cfba838d4dd479d9a0652b3e9a7f45c858d323))

* mention new run all script ([`916df28`](https://github.com/adamkarvonen/SAEBench/commit/916df289cdb0ad8565118b10bc3027cc249c6098))

* Script for running evals on all custom SAEs at once ([`b91c210`](https://github.com/adamkarvonen/SAEBench/commit/b91c2109047a1a544a5582c34620a87da01f16de))

* Rename formatting utils to general utils ([`dedec93`](https://github.com/adamkarvonen/SAEBench/commit/dedec939e87f9f8e459df2212f8ff130b29d063c))

* Clean up duplicated functions ([`28d2f2f`](https://github.com/adamkarvonen/SAEBench/commit/28d2f2f87da69237170bf4d6789de583ef6b340d))

* Clean up old graphing code ([`d3e8e87`](https://github.com/adamkarvonen/SAEBench/commit/d3e8e87b75e84d7d155ab95faa892b99a925d878))

* Fix memory leak ([`65fa76a`](https://github.com/adamkarvonen/SAEBench/commit/65fa76a934728ca263920d726bf9af3c4d9a7b6f))

* Make test file names consistent ([`70e2eaa`](https://github.com/adamkarvonen/SAEBench/commit/70e2eaa83328bae3128930357bdcb4ef820c3835))

* Remove unused flag ([`4ed9602`](https://github.com/adamkarvonen/SAEBench/commit/4ed96020ce8bb278b8380eebc05b2b5aeccea141))

* Improve GPU PCA training ([`01e6306`](https://github.com/adamkarvonen/SAEBench/commit/01e63064d9d905b47602fe123adca7167bde52e6))

* Fix namespace error and expected result format error ([`776e5f4`](https://github.com/adamkarvonen/SAEBench/commit/776e5f4a8cde70bbe74df322d605a4418da18796))

* Enable usage of core with custom SAEs ([`e359d1a`](https://github.com/adamkarvonen/SAEBench/commit/e359d1a1607612ad434a458bb9f017366e35517a))

* Add a function to fit the PCA using GPU and CUML ([`2632849`](https://github.com/adamkarvonen/SAEBench/commit/263284934db776e3f3368b369d35af22bae15f06))

* Switch from nested dict to list of tuples for selected_saes ([`dbdfe19`](https://github.com/adamkarvonen/SAEBench/commit/dbdfe1984c158e4b029a19cca85fffd5a220b2d8))

* Make it easier to train pca saes ([`eb41438`](https://github.com/adamkarvonen/SAEBench/commit/eb41438549033b574cb22cec583bae8bc106c875))

* Format with ruff ([`e62a436`](https://github.com/adamkarvonen/SAEBench/commit/e62a4368ea95afec390eecc7e43d7ef706426deb))

* Test identity SAE implementation ([`e95e055`](https://github.com/adamkarvonen/SAEBench/commit/e95e0556a94ccef41f1c826e5c9d54e033748a74))

* Add a PCA baseline ([`645a040`](https://github.com/adamkarvonen/SAEBench/commit/645a0405ecc42f671fc63697d3ecd811923d7702))

* Move unlearning tokenization function to general utils file, consolidate tokenization functions ([`98c4b5c`](https://github.com/adamkarvonen/SAEBench/commit/98c4b5cd4b31a37c61263e8813185b286e71659b))

* Merge pull request #34 from adamkarvonen/fix-core-eval-precision

fixing excessively low precision ([`e0ddf06`](https://github.com/adamkarvonen/SAEBench/commit/e0ddf06a96cc9a1e0f38bb23a77ef9fda92ef1dc))

* fixing excessively low precision ([`d1eea66`](https://github.com/adamkarvonen/SAEBench/commit/d1eea66f5e1e9af985922ca2552c6cfef85e4e43))

* Merge pull request #33 from adamkarvonen/add_baselines

Add baselines ([`20c2a40`](https://github.com/adamkarvonen/SAEBench/commit/20c2a40d2f672a94a82c7bd2ce9217c10845dfdc))

* Update README ([`e5b2ba4`](https://github.com/adamkarvonen/SAEBench/commit/e5b2ba43a50a6f92f30385f8910c65397399ae13))

* Fix regexes ([`478b41c`](https://github.com/adamkarvonen/SAEBench/commit/478b41c179264890ea8b2d938a81c4abed1a1ef6))

* Rename selection notebook ([`1b24a46`](https://github.com/adamkarvonen/SAEBench/commit/1b24a4663ba6b09461ecf376b99bd397acb94683))

* Remove usage of SAE patterns list ([`80ed74d`](https://github.com/adamkarvonen/SAEBench/commit/80ed74dfebce71d4d33643117d1e5f83ab41af51))

* Make sure batch is on the correct dtype ([`f8a1158`](https://github.com/adamkarvonen/SAEBench/commit/f8a11582b65a77b9547434862575b9797f45d811))

* Adapt auto interp to enable use with custom saes ([`7ea0e59`](https://github.com/adamkarvonen/SAEBench/commit/7ea0e590c02f8b43c0e8ca04aca09b2ecdb5fb94))

* Adapt absorption to match existing format ([`7e2ac58`](https://github.com/adamkarvonen/SAEBench/commit/7e2ac5833a05a9f468504d26c71d4d58528da4b9))

* Enable easy usage of evals with custom SAEs ([`64d2e23`](https://github.com/adamkarvonen/SAEBench/commit/64d2e2308d947ee0ac693d95064d1bcee43e5a42))

* Use sae.encode() for compatibility instead of sae.run_with_cache() ([`42cc9ce`](https://github.com/adamkarvonen/SAEBench/commit/42cc9ce2be47369df3eff2ef10913459afbdcde6))

* fix device errors ([`1725899`](https://github.com/adamkarvonen/SAEBench/commit/1725899024bb4933cc9f5ed282b82bda010fa454))

* format with ruff ([`737b788`](https://github.com/adamkarvonen/SAEBench/commit/737b7881428f1a21ef34d5409d0b08b602d4ef8c))

* Set autointerp context size in eval config ([`c4bfa82`](https://github.com/adamkarvonen/SAEBench/commit/c4bfa8230291bf57fc9dd6d0cd97ad4f48c36c0c))

* Add autointerp progress bars ([`bcc14a9`](https://github.com/adamkarvonen/SAEBench/commit/bcc14a9ccc689da7f97cac3d3bcc1e7eca9ec8cd))

* Use baseline SAEs on the sparse probing eval ([`d3e5e07`](https://github.com/adamkarvonen/SAEBench/commit/d3e5e0742f4522cf12e476c46bd97995651b18a0))

* Example jumprelu implementation ([`0ca103e`](https://github.com/adamkarvonen/SAEBench/commit/0ca103e5308c63a56e4760727d0460ceb63f4c26))

* identity sae baseline ([`5f65ace`](https://github.com/adamkarvonen/SAEBench/commit/5f65acead2625ce6ee5e34755377138fee03d349))

* Merge pull request #32 from adamkarvonen/core_evals_ignore_special

Added option to exclude special tokens from SAE reconstruction ([`186bdb4`](https://github.com/adamkarvonen/SAEBench/commit/186bdb4b20118cc8798fb50ef9172bde808c668f))

* Added option to exclude special tokens from SAE reconstruction ([`54a55f7`](https://github.com/adamkarvonen/SAEBench/commit/54a55f7f316a51be2640e2d162370465f3b49df2))

* Merge pull request #31 from adamkarvonen/activation_consolidation

Activation consolidation ([`3ddcceb`](https://github.com/adamkarvonen/SAEBench/commit/3ddccebb16eee7341c52bb9a476ffb856f8d4fe5))

* Add graphing for pythia and autointerp ([`1b4cf2a`](https://github.com/adamkarvonen/SAEBench/commit/1b4cf2a2fffc2354b68fe7fea02d9c33774bef4f))

* Correctly index into sae_acts ([`8c938ad`](https://github.com/adamkarvonen/SAEBench/commit/8c938ad8d78c15726a5a680d718ab7bb3a6c4f0d))

* Adapt format to Neuronpedia requirements ([`9baec7c`](https://github.com/adamkarvonen/SAEBench/commit/9baec7cab12d4228e709d259a933f0731a9552bd))

* Update README.md ([`de3ce5c`](https://github.com/adamkarvonen/SAEBench/commit/de3ce5c629bf844d5cc81f110c22f166b7ff6ff1))

* Rename for consistency ([`632f54d`](https://github.com/adamkarvonen/SAEBench/commit/632f54d63945423098c39a3a619c3c0f23ebcb88))

* Add end to end autointerp test ([`360dfe0`](https://github.com/adamkarvonen/SAEBench/commit/360dfe0b62a8cbad16113d809df84dd30285e0a5))

* Remove college biology from datasets as too close to wmdp_bio ([`dcdbbc5`](https://github.com/adamkarvonen/SAEBench/commit/dcdbbc5365cccb77edf38b77c2bff85339fef154))

* Print a warning if there aren&#39;t enough alive latents ([`23fc5a5`](https://github.com/adamkarvonen/SAEBench/commit/23fc5a59ea5cbb6f86be79dcf96eeeecf49a69c3))

* Include dataset info in filename ([`d37fc04`](https://github.com/adamkarvonen/SAEBench/commit/d37fc0497ce1e773c4c8b555c533e81e9de89edc))

* Add functions to encode precomputed activations ([`bd742ee`](https://github.com/adamkarvonen/SAEBench/commit/bd742eeec5adab41437af194359d998b63d783f3))

* Eliminate usage of activation store ([`fa58764`](https://github.com/adamkarvonen/SAEBench/commit/fa5876480dbcf52697ca658e51f54e4f4f316fc9))

* Adapt autointerp to new format ([`1de25f7`](https://github.com/adamkarvonen/SAEBench/commit/1de25f7f30350b3981f618b9833bbac966e69056))

* prepend bos token ([`46e00a4`](https://github.com/adamkarvonen/SAEBench/commit/46e00a4bad148bd3956b362c9ecdbd71dbef866f))

* Mask off BOS, EOS, and pad tokens ([`4e2b0d6`](https://github.com/adamkarvonen/SAEBench/commit/4e2b0d6c1b10b66941187d9fffe79f6c7a1e94f1))

* Collect the sparsity tensor for SAE autointerp ([`0dd3a91`](https://github.com/adamkarvonen/SAEBench/commit/0dd3a9169ee416484e23a8bb8934003caaee5c2d))

* Format with ruff ([`2afc772`](https://github.com/adamkarvonen/SAEBench/commit/2afc772d846fa0d5d620fd4341cbcfb8b030a9cc))

* Updated question ids running with one BOS token ([`df3b9d4`](https://github.com/adamkarvonen/SAEBench/commit/df3b9d40ad273debd64801e84bfc69c4cecc8143))

* Zero out SAE activations on BOS token ([`0d30360`](https://github.com/adamkarvonen/SAEBench/commit/0d30360622a81c20dd2b86cd470e6e0ca0376612))

* Only use one BOS token at beginning ([`ad97556`](https://github.com/adamkarvonen/SAEBench/commit/ad97556770d6b99cfa8e2d7d2caa336d426e538b))

* Remove redundant with no_grad() ([`528959f`](https://github.com/adamkarvonen/SAEBench/commit/528959f1746a5185ccb3e69cce7ddc6d32466c34))

* Merge remote-tracking branch &#39;origin/main&#39; into activation_consolidation ([`777e9d4`](https://github.com/adamkarvonen/SAEBench/commit/777e9d4de41c5eb7a77002c9ea32ff7046d70921))

* Merge pull request #8 from callummcdougall/callum/autointerp

Autointerp eval ([`6f81495`](https://github.com/adamkarvonen/SAEBench/commit/6f814954cb39be1ac3f54ca14e4a79876e23eaf0))

* Merge branch &#39;main&#39; into callum/autointerp ([`466a37d`](https://github.com/adamkarvonen/SAEBench/commit/466a37d742bfa8adfe8e79284e3ca1d005463aba))

* fix device bug, resolve test utils conflicts ([`9b2e909`](https://github.com/adamkarvonen/SAEBench/commit/9b2e9091adfd3e1b8f3e933e0ab2717ee0fb67f6))

* fix ctx len bug, handle dead features better ([`63c2561`](https://github.com/adamkarvonen/SAEBench/commit/63c256119f6fc1f555c385a18977a031347e78ed))

* fix topk error ([`e39a9ab`](https://github.com/adamkarvonen/SAEBench/commit/e39a9aba072298e16cf309f4efca22c2bf3cbedb))

* add sae encode function ([`b247f91`](https://github.com/adamkarvonen/SAEBench/commit/b247f91234929bffa5ee79a08e265f57f5222095))

* first commit ([`4b23575`](https://github.com/adamkarvonen/SAEBench/commit/4b23575cca0f9422c7cdbefac093c6dead6a55eb))

* Move the get_sparsity function to general utils folder, mask bos, pad, and eos tokens for unlearning ([`f679f0f`](https://github.com/adamkarvonen/SAEBench/commit/f679f0fd6944bef1edc048c01cefaeccaaee7458))

* Make it easier to use get_llm_activations() with other evals ([`1ed9a29`](https://github.com/adamkarvonen/SAEBench/commit/1ed9a297f974a4916881fb82026835e3ce1c2102))

* Improve graphing notebook for current output format ([`36fb3ba`](https://github.com/adamkarvonen/SAEBench/commit/36fb3ba97fe1b31ccd9c8233e7c4e5b99b548f60))

* Apply nbstripout ([`ca27e41`](https://github.com/adamkarvonen/SAEBench/commit/ca27e415355ecc91c5a4ec56ed1257f890a1de46))

* Notebook specifically for graphing and analyzing mdl results ([`114cefb`](https://github.com/adamkarvonen/SAEBench/commit/114cefbd982bceb6c656cb6ddef714a53b94d37b))

* Merge pull request #30 from adamkarvonen/mdl_fixes

Mdl fixes ([`65c3c98`](https://github.com/adamkarvonen/SAEBench/commit/65c3c9873ea0b2820457218ac3c43b88ebd31a18))

* Add example data and add details to README. ([`21f3c83`](https://github.com/adamkarvonen/SAEBench/commit/21f3c8358581dc13e24020d52527215a8c7705e3))

* Use torch instead of t for consistency ([`903324f`](https://github.com/adamkarvonen/SAEBench/commit/903324f60b87ee6210ab5c63c8cf45bcc329a2b1))

* Move calculations to float32 to avoid dtype errors ([`099a94f`](https://github.com/adamkarvonen/SAEBench/commit/099a94f4bbefa96e5d7848de5c434b3fe57747c3))

* Further general cleanup of mdl eval ([`51e9b60`](https://github.com/adamkarvonen/SAEBench/commit/51e9b6025b59034b63e239011955a19d8d8570a7))

* Save sae results per sae ([`7045ad2`](https://github.com/adamkarvonen/SAEBench/commit/7045ad23e671f457e961a52e8a888cec5dd3d5b1))

* Fix variable name bug ([`d5ec2d2`](https://github.com/adamkarvonen/SAEBench/commit/d5ec2d294d38a0db636623c7d02969d80cbd7b99))

* MDL is running ([`c486454`](https://github.com/adamkarvonen/SAEBench/commit/c4864547028a9e0f267c9f9e52d2bc7bd9e036cf))

* Format with ruff ([`04b14b7`](https://github.com/adamkarvonen/SAEBench/commit/04b14b76cf11e7b10e9b94898804bd5f3ed468a9))

* Add descriptions to unlearning hyperparameters and descriptions of shift, tpp, and sparse probing evals. ([`6c141c1`](https://github.com/adamkarvonen/SAEBench/commit/6c141c10ff501c52512304c38fb46ff154c6905b))

* Merge pull request #28 from adamkarvonen/update_unlearning

Update unlearning output format ([`2bfd70b`](https://github.com/adamkarvonen/SAEBench/commit/2bfd70bd1e1814f9f58d1416b6cf8539ee787ecf))

* descriptions ([`c1f79b3`](https://github.com/adamkarvonen/SAEBench/commit/c1f79b3b348db4ca5aeab99b271e8a9d7b840603))

* Update ([`60247d0`](https://github.com/adamkarvonen/SAEBench/commit/60247d08906d286c9de5191523ef39533edb30a1))

* update description ([`03a2402`](https://github.com/adamkarvonen/SAEBench/commit/03a2402e8ef47301be49f2258af8db21e5ce10b1))

* fix unlearning test ([`7c50173`](https://github.com/adamkarvonen/SAEBench/commit/7c50173f2424af0906c423d336cef158a206a12a))

* remove artifact ([`41c7750`](https://github.com/adamkarvonen/SAEBench/commit/41c7750c04f43201c05c309da19ce0b19f11161c))

* output format ([`1201a7c`](https://github.com/adamkarvonen/SAEBench/commit/1201a7cd89717a35ff2a04af244a7b763a900510))

* update unlearning test ([`1eae76b`](https://github.com/adamkarvonen/SAEBench/commit/1eae76b26b7bd54be1db06b101701aef82ab54d0))

* unlearning start ([`a7be6df`](https://github.com/adamkarvonen/SAEBench/commit/a7be6dfe4161aa932e4a68f1757acbe5b0bb9253))

* Merge pull request #27 from adamkarvonen/core_tests

Update JSON schema filenames ([`b6ed053`](https://github.com/adamkarvonen/SAEBench/commit/b6ed053cb5abd3ee9757da9d74a0a7adf0244019))

* remove unused ([`810afe8`](https://github.com/adamkarvonen/SAEBench/commit/810afe8acebcc7df744aff72a89fb0b3d3d8256d))

* updated schema file names ([`e4df309`](https://github.com/adamkarvonen/SAEBench/commit/e4df3090bc3ffb76bbd7b02b9191eebd653c52ad))

* update name of output schema file ([`af58f0f`](https://github.com/adamkarvonen/SAEBench/commit/af58f0f8bd9a91e69061a97e35bbffc987e37d5b))

* Merge pull request #26 from adamkarvonen/core_tests

Update ui_default_display, titles for display ([`371b80d`](https://github.com/adamkarvonen/SAEBench/commit/371b80d60427cccd551f711cbcee5b6e445d0ab3))

* Update titles ([`4885187`](https://github.com/adamkarvonen/SAEBench/commit/4885187ff147439290a9d342cb0f4b1a8219bc7e))

* default display ([`8f811b8`](https://github.com/adamkarvonen/SAEBench/commit/8f811b899a6e6720952c5941f6ccb2964914a55e))

* Merge pull request #25 from adamkarvonen/core_tests

added tests for core eval output ([`92bc76a`](https://github.com/adamkarvonen/SAEBench/commit/92bc76a582e0980c4a23a0280ec944795cace800))

* added tests for core eval output ([`591ce3f`](https://github.com/adamkarvonen/SAEBench/commit/591ce3f58e3546b7495c31f1d3dd20ce25724c3f))

* Add end to end unlearning test ([`79435ab`](https://github.com/adamkarvonen/SAEBench/commit/79435ab0ae8c9873a1d98d44f6963eeb487d1872))

* clean up activations always defaults to false ([`5d545d1`](https://github.com/adamkarvonen/SAEBench/commit/5d545d199309334ac9985c036cbfb3b2f1a4638d))

* Merge pull request #24 from adamkarvonen/core_update

New Core output format, plus converter ([`b01be8a`](https://github.com/adamkarvonen/SAEBench/commit/b01be8a150e69e95beecf5b781f1ed8d2183b248))

* New Core output format, plus converter ([`192e92b`](https://github.com/adamkarvonen/SAEBench/commit/192e92b12678c2784a6a33b498072a90eb94c6a6))

* Merge pull request #6 from koayon/mdl-eval

Implement MDL eval ([`829de0c`](https://github.com/adamkarvonen/SAEBench/commit/829de0caa7d7bd67ba5ec86595ef4ab82b825668))

* Merge branch &#39;main&#39; into mdl-eval ([`ad18568`](https://github.com/adamkarvonen/SAEBench/commit/ad185689f403d3d4c0a036b23fb1b420d0ee2151))

* Generate bfloat16 question_ids and commit them to the proper location, remove old ones ([`d48c68b`](https://github.com/adamkarvonen/SAEBench/commit/d48c68b707f3c2aef669b14d4e841adaea8bb2ca))

* Add example unlearning output ([`9866453`](https://github.com/adamkarvonen/SAEBench/commit/9866453230a6457612b6b1cdb0fefd0e28ff5e31))

* Merge pull request #23 from adamkarvonen/unlearning_adapt

Unlearning adapt ([`3c48cdb`](https://github.com/adamkarvonen/SAEBench/commit/3c48cdb6ed0bd3a1b7b0f733561abaa1c841b0ce))

* Allow plotting of gemma SAEs ([`c813b39`](https://github.com/adamkarvonen/SAEBench/commit/c813b39824f567858c93cce4db5338b755ef21de))

* Adapt unlearning eval to new format ([`26d1675`](https://github.com/adamkarvonen/SAEBench/commit/26d167512c4a5804b433e405b9e611702afdd9f7))

* pass artifact folder in to unlearning functions ([`87508a6`](https://github.com/adamkarvonen/SAEBench/commit/87508a6cbcbcbb743caf39536fcf6ceb8dcca8cf))

* Add a sparsity penalty when training the SHIFT / TPP linear probes ([`cc73c6f`](https://github.com/adamkarvonen/SAEBench/commit/cc73c6f1f0123d66fa562c4b446267504e605fb7))

* Merge pull request #21 from adamkarvonen/shift_sparse_probing_descriptions

Shift sparse probing descriptions ([`3e9555a`](https://github.com/adamkarvonen/SAEBench/commit/3e9555a5fa48a0ec0c4cd9612f3f0fbbe994457b))

* Remove unnecessary test keys, add note to README ([`12f324d`](https://github.com/adamkarvonen/SAEBench/commit/12f324d66ba3667d055c788436e4c92d2d79c3cb))

* Finish rename of the spurious_corr variable ([`b28aab6`](https://github.com/adamkarvonen/SAEBench/commit/b28aab68045466594cd8328312371b9b8c8f02e4))

* update doc about how to update json schemas files. add json schema files. ([`95fda67`](https://github.com/adamkarvonen/SAEBench/commit/95fda67f02899e53a4d43aa62d1e611b3b0f5f58))

* Update from uncategorized to shift_metrics and tpp_metrics ([`43bf1f4`](https://github.com/adamkarvonen/SAEBench/commit/43bf1f40d345d1dd01245d64a5d8441223c6c827))

* Improve titles and descriptions for sparse probing ([`177be38`](https://github.com/adamkarvonen/SAEBench/commit/177be38c320cec0cf2e5ceb66d94e6e5c80427c6))

* Improve descriptions, titles, and variable names in SHIFT and TPP ([`29e1ecc`](https://github.com/adamkarvonen/SAEBench/commit/29e1ecc5675487aa8d027c563ad0fce7ed93aaf0))

* Merge pull request #22 from adamkarvonen/fix/handle_gated_in_core

handle case where gated SAEs don&#39;t have b_enc ([`586597b`](https://github.com/adamkarvonen/SAEBench/commit/586597b61697da55142844b9658977d7b14fbf32))

* handle case where gated SAEs don&#39;t have b_enc ([`355aaf4`](https://github.com/adamkarvonen/SAEBench/commit/355aaf4d666705d6a377d5872a75e4748f2bad5b))

* Merge pull request #20 from adamkarvonen/make_unstructured_optional

fix: eval_result_unstructured should be optional ([`76d72a6`](https://github.com/adamkarvonen/SAEBench/commit/76d72a6d4af58d80b805dc51ababec1318afa954))

* Merge pull request #19 from adamkarvonen/core_eval_incremental_saving

Core eval incremental saving ([`7ddd55a`](https://github.com/adamkarvonen/SAEBench/commit/7ddd55ad690b8563942d0cfa126d5d4feb88c9f0))

* added error handling and exponential backoff ([`92abbbe`](https://github.com/adamkarvonen/SAEBench/commit/92abbbeae83ebea713a03e22bb009b1b0a9af906))

* added code to produce intermediate json output between SAEs ([`b15a2e2`](https://github.com/adamkarvonen/SAEBench/commit/b15a2e2297a9d574cbd87b5bde62d1ad7c443aef))

* Merge pull request #18 from adamkarvonen/set_sparse_probing_default_display

set k = 1, 2, 5 default display = true for sparse probing ([`db93af6`](https://github.com/adamkarvonen/SAEBench/commit/db93af64d208b6ff309f3d7b3b288742ac5b8248))

* set k = 1, 2, 5 default display = true for sparse probing ([`bf9b5ac`](https://github.com/adamkarvonen/SAEBench/commit/bf9b5ac68e36dde16d4c88983b3018789b9167a4))

* Merge pull request #17 from adamkarvonen/add_unstructured_eval_output

Feature: Support unstructured eval output ([`3b17927`](https://github.com/adamkarvonen/SAEBench/commit/3b179275f1419f71a924b60609dea29b83e44f7a))

* Support unstructured eval output ([`adf028a`](https://github.com/adamkarvonen/SAEBench/commit/adf028a0352867e4aaebcc2e8a76c137fd21f9bd))

* Merge pull request #16 from adamkarvonen/basic-evals

Added core evals to repo ([`c55f48f`](https://github.com/adamkarvonen/SAEBench/commit/c55f48ffa4dc45c677f8c133a2f9638ee40dd444))

* Added core evals to repo ([`9b1dd45`](https://github.com/adamkarvonen/SAEBench/commit/9b1dd45af4ae0b67e69a73e4e1d87d64976dd0cc))

* Merge pull request #15 from adamkarvonen/json_schema_absorption

Use Pydantic for eval configs and outputs for annotations and portability ([`e75c8b5`](https://github.com/adamkarvonen/SAEBench/commit/e75c8b5e6412bc59ce5bfeae5bfffae9a330869b))

* update shift/tpp and sparse probing to evaloutput format ([`2dbb6f8`](https://github.com/adamkarvonen/SAEBench/commit/2dbb6f81faa9dfff20010a8458de9219bf8c7a16))

* Merge remote-tracking branch &#39;origin/main&#39; into json_schema_absorption ([`eb8c660`](https://github.com/adamkarvonen/SAEBench/commit/eb8c66073524128b3154051fe47f8fe3ef85959a))

* Add pytorch cuda flag due to OOM error message ([`c8e74f4`](https://github.com/adamkarvonen/SAEBench/commit/c8e74f46869d50a313c24e6b54f9dd4153bb8098))

* confirm shift_and_tpp to new output format ([`153c713`](https://github.com/adamkarvonen/SAEBench/commit/153c713d8fa8c68e188e52f6b04b67d01f6c3931))

* Merge remote-tracking branch &#39;origin/main&#39; into json_schema_absorption ([`b337d5f`](https://github.com/adamkarvonen/SAEBench/commit/b337d5f762ab5ea729f7e151e4b9e6b5d42a1a32))

* Add example regexes for gemma 2 2b ([`aea66aa`](https://github.com/adamkarvonen/SAEBench/commit/aea66aa3cb9b1cee84e85a050fca7fbc341aca3b))

* Merge pull request #14 from adamkarvonen/shift_sparse_probing_updates

Shift sparse probing updates ([`14b5025`](https://github.com/adamkarvonen/SAEBench/commit/14b5025de982a1dc1789a679a4a1ac456692d7c8))

* Add example usage of gemma-scope and gemma SAEs ([`f2dcacf`](https://github.com/adamkarvonen/SAEBench/commit/f2dcacfc82ae50228c4957ec1a368332fe0cc2d4))

* Improve arg parsing and probe file name ([`ec8cd87`](https://github.com/adamkarvonen/SAEBench/commit/ec8cd872060ef158ca0ec0ef5e8873546fadeff4))

* Mention other use for GPU probe training ([`929cdc0`](https://github.com/adamkarvonen/SAEBench/commit/929cdc08e86342c13800d496cb463b70aca61cd7))

* Add early stopping patience to reduce variance ([`5285563`](https://github.com/adamkarvonen/SAEBench/commit/5285563578bd4d00471e50eb91cec96bf310d33d))

* Add note on random seed being overwritten by argparse ([`d5a215a`](https://github.com/adamkarvonen/SAEBench/commit/d5a215ae3c1b7a33e1632670822514df2041d741))

* Separate save areas for tpp and shift ([`c860deb`](https://github.com/adamkarvonen/SAEBench/commit/c860deba41d00cccc6bfdd2ebebcb31328d2a794))

* Also ignore artifacts and test results ([`1a81702`](https://github.com/adamkarvonen/SAEBench/commit/1a8170258f05b101a0fe877f9f1839d139b607ee))

* Add shift and tpp to new format ([`d7e4b8b`](https://github.com/adamkarvonen/SAEBench/commit/d7e4b8b8cf6f28799cc95b6bb190340505dda72b))

* Improve assert error message if keys don&#39;t match ([`c514f49`](https://github.com/adamkarvonen/SAEBench/commit/c514f49c302a9a7c19c7bfa526d7197c575c58a8))

* force_rerun now reruns even if a results file exists for a given sae ([`1ff0b16`](https://github.com/adamkarvonen/SAEBench/commit/1ff0b1678b51bafd4c59dbf44824e4046122ccb1))

* Make shift and tpp tests compatible with old results ([`ff2f46d`](https://github.com/adamkarvonen/SAEBench/commit/ff2f46d3dd61d6cbeaba34c265f5f5b5dd1a7e70))

* Make sparse probing test backwards compatible with old results ([`782a080`](https://github.com/adamkarvonen/SAEBench/commit/782a0806b800fc71e1c030fbc7c62fca2d85b9e3))

* Create a new graphing notebook for regex based selection ([`65ef605`](https://github.com/adamkarvonen/SAEBench/commit/65ef605551e00773095680bb956994c306ec146c))

* Improve artifacts and results storage locations, add a utility to select saes using multiple regex patterns ([`1cdfdb7`](https://github.com/adamkarvonen/SAEBench/commit/1cdfdb712f3a7a8c51d21611817652889b9cb33a))

* No longer aggregate over saes in a dict ([`0e5bccb`](https://github.com/adamkarvonen/SAEBench/commit/0e5bccb4fa57a34c6c92294312a8764257993f56))

* Rename old graphing file ([`df72d30`](https://github.com/adamkarvonen/SAEBench/commit/df72d30497834a8f1ea259e2118f9c1a0845a749))

* Add openai and tabulate to requirements.txt ([`f626447`](https://github.com/adamkarvonen/SAEBench/commit/f626447452d4e0e9bb9286959b73cbd8fa138e83))

* Begin shift / tpp adaptation ([`ab1f062`](https://github.com/adamkarvonen/SAEBench/commit/ab1f062a935dc33fcbec6aa951ad88c68b4ebd5e))

* No longer average over multiple saes ([`aaf06eb`](https://github.com/adamkarvonen/SAEBench/commit/aaf06ebffd16479ff021ddd62f46d3b8b756f6c2))

* Add an optional list of regexes ([`41b86a4`](https://github.com/adamkarvonen/SAEBench/commit/41b86a4cc0f029303e4ed127f3f869e6f23a4f54))

* By default remove the bos token ([`4877424`](https://github.com/adamkarvonen/SAEBench/commit/4877424bdd97b1c9a6ded7cef15c278897351f68))

* Match new sae bench format ([`5d484e3`](https://github.com/adamkarvonen/SAEBench/commit/5d484e33542a287077718da60283e6c40e82f0f1))

* Add note on output formats ([`56c637f`](https://github.com/adamkarvonen/SAEBench/commit/56c637f7b0679e70c8d8acbaedfffd4778f5b5e7))

* Add notes on custom sae usage ([`30d4f16`](https://github.com/adamkarvonen/SAEBench/commit/30d4f165f96f48d2c7b8bfe856c2cbac4c16801f))

* Add a utility function to plot multiple results at once ([`a89c86e`](https://github.com/adamkarvonen/SAEBench/commit/a89c86e38e4a8f8f83d55b6b6a4dc10a680164c8))

* Ignore images and results folders ([`e07e65f`](https://github.com/adamkarvonen/SAEBench/commit/e07e65fad7c7373489326b1df10f01e55d6c1a1e))

* test pre-commit hook ([`648046f`](https://github.com/adamkarvonen/SAEBench/commit/648046ff40cc66b9e471512de339cc7d963ea0a6))

* produce the JSON schema files and add as a pre-commit hook ([`6683e17`](https://github.com/adamkarvonen/SAEBench/commit/6683e17cc21c85edc088126ef57ebd2d80129a5c))

* fix absorption test ([`d62b752`](https://github.com/adamkarvonen/SAEBench/commit/d62b752360ad4d0957234066566744b472241390))

* don&#39;t commit artifact file ([`2c5691a`](https://github.com/adamkarvonen/SAEBench/commit/2c5691a97f698454bc50ce8d97712231426fbe54))

* Merge branch &#39;main&#39; into mdl-eval ([`922fb14`](https://github.com/adamkarvonen/SAEBench/commit/922fb142aa262ffca815f92cc59651f7624acf2c))

* Merge pull request #12 from jbloomAus/demo-format-and-command-changes-absorption

Demo of Changes to enable easy running of evals at scale (using absorption) ([`a9603b8`](https://github.com/adamkarvonen/SAEBench/commit/a9603b82df8a370957f0d7a30dbdb02ad207a361))

* Merge branch &#39;main&#39; into demo-format-and-command-changes-absorption ([`9a9d4b1`](https://github.com/adamkarvonen/SAEBench/commit/9a9d4b1197c33c4f22c224e0469b524127a331f9))

* Merge pull request #13 from adamkarvonen/minor_shift_improvements

Minor shift improvements ([`1b318d5`](https://github.com/adamkarvonen/SAEBench/commit/1b318d5d30ad1c3a9748a000062daaf431689a54))

* Notebook used to test different datasets ([`d4d4fb5`](https://github.com/adamkarvonen/SAEBench/commit/d4d4fb58db72d6b7d50bf67f9e9b25fb2e3cb891))

* Comment out outdated tests ([`f1e2f9e`](https://github.com/adamkarvonen/SAEBench/commit/f1e2f9e65b41336d283edb6a4f69a550b61bde5a))

* Add runtime estimates to READMEs ([`32da7aa`](https://github.com/adamkarvonen/SAEBench/commit/32da7aaba40d0e6b6065900c43d204a6756410e6))

* Rename to match other readmes ([`8dc952f`](https://github.com/adamkarvonen/SAEBench/commit/8dc952f9db8343a10720a4c5701407cf4d9fc055))

* Reduce the default amount of n values for faster runtime ([`74fd9a1`](https://github.com/adamkarvonen/SAEBench/commit/74fd9a150bc577a41fd99830ef76ebc9113ab912))

* Lower peak memory usage to fit on a 3090 ([`1fecf15`](https://github.com/adamkarvonen/SAEBench/commit/1fecf156fb3ff774c8c1fc42151746258634a6d2))

* Skip first 150 chars per Neurons in a Haystack ([`46d9510`](https://github.com/adamkarvonen/SAEBench/commit/46d9510dd7dbb1aa9c3bd61111cf55e52d6425fe))

* Merge pull request #11 from adamkarvonen/add_datasets

Add additional sparse probing datasets ([`0512456`](https://github.com/adamkarvonen/SAEBench/commit/0512456f189f5f9e775f763114b33013a09213d9))

* Share dataset creation code between tpp and sparse probing evals ([`30f60b6`](https://github.com/adamkarvonen/SAEBench/commit/30f60b6ea6015c4b32d2bcbf1fd448ed36b3584b))

* Update scr and tpp tests ([`8561e91`](https://github.com/adamkarvonen/SAEBench/commit/8561e91b896a533dae01080db7b2107c8964305e))

* Add an optional keys to compare list, to only compare those values ([`d60388b`](https://github.com/adamkarvonen/SAEBench/commit/d60388baab5709a4043bbc42a24ff2075ab99870))

* Add ag_news and europarl datasets ([`82de70f`](https://github.com/adamkarvonen/SAEBench/commit/82de70f57daa3aab5ad39266dc847242f311c04a))

* Add a single shift scr metric key ([`b65f969`](https://github.com/adamkarvonen/SAEBench/commit/b65f969d389a0ddc5c32d33fddb1dbac648b5ef4))

* Use new sparse probing dataset names ([`7b32c83`](https://github.com/adamkarvonen/SAEBench/commit/7b32c83529e6decc4213013de4cc45e183b82b53))

* Use more probe epochs, update to use new dataset names ([`b49db06`](https://github.com/adamkarvonen/SAEBench/commit/b49db06a8fdd4d67ca21eb71b8efe6c2d15f3c3e))

* Add several new sparse probing datasets ([`b4f5400`](https://github.com/adamkarvonen/SAEBench/commit/b4f5400badfbc2703422109df8f2b61bb95d9328))

* Add dataset functions for amazon sentiment and github code ([`aa1a478`](https://github.com/adamkarvonen/SAEBench/commit/aa1a47878e8bb300fd769d248fd2909634965538))

* Use full huggingface dataset names ([`d2d4001`](https://github.com/adamkarvonen/SAEBench/commit/d2d4001497a075b68026bff753966a12f9ab4515))

* Change default unlearning hyperparameters ([`d4c1949`](https://github.com/adamkarvonen/SAEBench/commit/d4c19495b761d89fed1ccdaf12a9a88d69dc54ad))

* Do further analysis of unlearning hyperparameters ([`8f6262c`](https://github.com/adamkarvonen/SAEBench/commit/8f6262ccb8ef5e259b61ca94e994975809057702))

* Add multiple subsets of existing datasets ([`ae36e81`](https://github.com/adamkarvonen/SAEBench/commit/ae36e81d5fbec11a2494193d79880e064c967805))

* Retry loading dataset due to intermittent errors ([`dc38fd0`](https://github.com/adamkarvonen/SAEBench/commit/dc38fd06ee2057cef39dead9ee56302e741e1e91))

* Use stop at layer for faster inference ([`e752a9c`](https://github.com/adamkarvonen/SAEBench/commit/e752a9cf8e04a93998aca18cb6c7ee4ed0f1c946))

* Merge pull request #10 from curt-tigges/main

Initial RAVEL code ([`fc6a59b`](https://github.com/adamkarvonen/SAEBench/commit/fc6a59baa414a95fe4529b0fb067fbd398a20c6e))

* Merge branch &#39;main&#39; into main ([`d132ec3`](https://github.com/adamkarvonen/SAEBench/commit/d132ec3f63b22771cc6e519b7599fcd606d07363))

* Merge pull request #9 from adamkarvonen/unlearning_cleanup

Unlearning cleanup ([`209e526`](https://github.com/adamkarvonen/SAEBench/commit/209e52601047e0eff2d3f887b52ed6bd6c6ed92d))

* Get known question ids if they don&#39;t exist ([`f3516f4`](https://github.com/adamkarvonen/SAEBench/commit/f3516f49680723c5fa00ef1c470d841169eb9047))

* Remove unused functions ([`ac3da4d`](https://github.com/adamkarvonen/SAEBench/commit/ac3da4d02d17fc297a527c93b41892e90d62b580))

* discard unused variable ([`aea5531`](https://github.com/adamkarvonen/SAEBench/commit/aea55316cb9855c169247f1c1a2d85584939e38b))

* Get results for all retain thresholds ([`08eec18`](https://github.com/adamkarvonen/SAEBench/commit/08eec18c88cab748df9f7e45f9aabb08a0beabe9))

* Save unlearning score in final output ([`f90b114`](https://github.com/adamkarvonen/SAEBench/commit/f90b114f86fb20864e0d09b02b2269841cd504a5))

* Add file to get correct answers for a model ([`0361c07`](https://github.com/adamkarvonen/SAEBench/commit/0361c07ac29a5d893b08bca5e115099450006757))

* Fix missing filenames ([`6aad3ca`](https://github.com/adamkarvonen/SAEBench/commit/6aad3ca402e1e28e36a939c529f81dfebb976bdb))

* Move hyperparameters to eval config ([`5d2b9d0`](https://github.com/adamkarvonen/SAEBench/commit/5d2b9d050c96527854e367884ed8bedb558c50ce))

* Move llm and sae to llm_dtype ([`56e8e43`](https://github.com/adamkarvonen/SAEBench/commit/56e8e43419d171e3cd9b943f581122b782fbded3))

* Fix utils import ([`79321f8`](https://github.com/adamkarvonen/SAEBench/commit/79321f897f6f9344c07bfb2c7b9201deebba5b09))

* Apply ruff formatter ([`1423c33`](https://github.com/adamkarvonen/SAEBench/commit/1423c33f419d4b5d036142fe8034a84580f1a9dc))

* Make sure we don&#39;t commit the forget dataset ([`63bc153`](https://github.com/adamkarvonen/SAEBench/commit/63bc1535323caed4def2f8cb628ca9facb4b7ae8))

* Apply nbstripout ([`2ff3e72`](https://github.com/adamkarvonen/SAEBench/commit/2ff3e72955a180da880ec128311c036fb0429cf6))

* Merge pull request #7 from yeutong/unlearning

implement unlearning eval ([`42de6df`](https://github.com/adamkarvonen/SAEBench/commit/42de6df06f23c69b732d5733806b7d72414f1c70))

* Merge branch &#39;main&#39; into unlearning ([`b2f6d68`](https://github.com/adamkarvonen/SAEBench/commit/b2f6d6880059170dc210a50066a7b98ac0c1e616))

* Merge remote-tracking branch &#39;upstream/main&#39; ([`daab3e2`](https://github.com/adamkarvonen/SAEBench/commit/daab3e25fbb77ae8b1920ba383338280a4ddacae))

* add analysis ([`5eb7dfa`](https://github.com/adamkarvonen/SAEBench/commit/5eb7dfa94a04b36f8649986034e790cccf6e2617))

* success ([`b67c97b`](https://github.com/adamkarvonen/SAEBench/commit/b67c97b2690f18d2ecdf060428de18ebd2c3211e))

* add gemma-2-2b-it ([`5f502d7`](https://github.com/adamkarvonen/SAEBench/commit/5f502d7b46628f8f710f6bb415acb38c712a3da1))

* revert changes to template.ipynb ([`aac04e0`](https://github.com/adamkarvonen/SAEBench/commit/aac04e0e5fc8bb12395f3b431a55a386ebeffcbd))

* fix detail ([`e6e0985`](https://github.com/adamkarvonen/SAEBench/commit/e6e0985b70087d4e8a93d60613c70f82fdcf8302))

* arrange structure ([`6392064`](https://github.com/adamkarvonen/SAEBench/commit/63920643775f2a5e5fd1827af8536f90b5c5ae8b))

* replace model and sae loading ([`91ce2fd`](https://github.com/adamkarvonen/SAEBench/commit/91ce2fd9088c7dbb8063ba24556b1407d3942da2))

* moved all code ([`7a714b4`](https://github.com/adamkarvonen/SAEBench/commit/7a714b41413866aa40f0919f82aea16101fecca7))

* Updated notebook ([`ae40301`](https://github.com/adamkarvonen/SAEBench/commit/ae403013856e8ce13da4bb6c3e5df3ca76254774))

* Added initial demo notebook ([`86dfd95`](https://github.com/adamkarvonen/SAEBench/commit/86dfd95e4bac9acdde9a37a511941703beaa8987))

* Added initial RAVEL files for dataset generation ([`96e963f`](https://github.com/adamkarvonen/SAEBench/commit/96e963fe73cdea64dd6b127f0440545b6938a0fc))

* delete old template ([`0092a1f`](https://github.com/adamkarvonen/SAEBench/commit/0092a1fc49fd18ac1315b18b3a99d778be02d8a5))

* add re-usable testing utils for the config, cli and output format. ([`03aee86`](https://github.com/adamkarvonen/SAEBench/commit/03aee86846ea7e83c7f885ebe7ec0f33ce908960))

* delete old template ([`8d66a49`](https://github.com/adamkarvonen/SAEBench/commit/8d66a4978360545d69a907e680dbd3fd1a5fa9a8))

* update stategy for running absorption via CLI ([`13f90d0`](https://github.com/adamkarvonen/SAEBench/commit/13f90d0647a0694112a82c1820b789fdcdfa5108))

* add regex based sae selection strategy ([`57e9be0`](https://github.com/adamkarvonen/SAEBench/commit/57e9be0ac9199dba6b9f87fe92f80532e9aefced))

* restructure results json, store probe results ([`9ec59a8`](https://github.com/adamkarvonen/SAEBench/commit/9ec59a85fd9783387e0b8f904aca157143092cd2))

* add version control utils ([`b516958`](https://github.com/adamkarvonen/SAEBench/commit/b5169583a8a150bfb95d6dae9df9df29eab6f153))

* Add pytorch flag due to CUDA OOM message ([`c57eef7`](https://github.com/adamkarvonen/SAEBench/commit/c57eef755554220309270583f19ed7816e978074))

* Move sae to llm dtype ([`c510d95`](https://github.com/adamkarvonen/SAEBench/commit/c510d95e4f1b7198166d7869b1644b7389e7143e))

* Add a README and test for absorption ([`a8f4190`](https://github.com/adamkarvonen/SAEBench/commit/a8f41907cd9a0d122df1ca72e80e8b7ea3cede28))

* Add example main function to absorption eval ([`2f1c551`](https://github.com/adamkarvonen/SAEBench/commit/2f1c551283dad9448b13a2862190f16ee3675e78))

* Move sae to llm dtype ([`26ed7a0`](https://github.com/adamkarvonen/SAEBench/commit/26ed7a0a63692563892e167af85d2588ce3569d2))

* Merge pull request #3 from chanind/absorption

Feature Absorption Eval ([`8d80be6`](https://github.com/adamkarvonen/SAEBench/commit/8d80be6f6beacc42c706185ac0f9828a9d109865))

* renaming dict keys ([`ff81c53`](https://github.com/adamkarvonen/SAEBench/commit/ff81c53d7652b121b4376bdc88240aa80da9b368))

* fixing batching error in absorption calculator ([`8a425cd`](https://github.com/adamkarvonen/SAEBench/commit/8a425cdf2a33c50985ae55ac2d7261c0a3c58a1f))

* Merge branch &#39;main&#39; into absorption ([`822ad11`](https://github.com/adamkarvonen/SAEBench/commit/822ad11d578789e7f32993fbf24e450dda9a5073))

* Merge pull request #5 from koayon/rename-utils

Rename utils to avoid name conflict ([`eb6cc7b`](https://github.com/adamkarvonen/SAEBench/commit/eb6cc7b920cfeffec9cad7ea82dadaa28a2c03f0))

* update notebook imports ([`6b4ca4a`](https://github.com/adamkarvonen/SAEBench/commit/6b4ca4a154e88d9f52137214af9c374e144514cb))

* notebook reversion ([`1391c79`](https://github.com/adamkarvonen/SAEBench/commit/1391c79c58672e7178bfa04c61d2d59fa661b539))

* indentation ([`111a68b`](https://github.com/adamkarvonen/SAEBench/commit/111a68b43df1d1c95f8f89a72c90c6b98e55754c))

* rename utils to avoid name conflict ([`e2a380a`](https://github.com/adamkarvonen/SAEBench/commit/e2a380a063d3d58bd2827456a6219305ef184841))

* Merge branch &#39;main&#39; into absorption ([`5db6ecb`](https://github.com/adamkarvonen/SAEBench/commit/5db6ecb2591fbb9462c5f2b4bd629a7841c9cea5))

* reverting to original sparse probing main.py ([`8d6779f`](https://github.com/adamkarvonen/SAEBench/commit/8d6779f1f80569a5faa27614582e236de88e9352))

* fixing dtypes ([`280f51a`](https://github.com/adamkarvonen/SAEBench/commit/280f51aadb9ad4c647b540a2876a2826f0c58c58))

* adding absorption calculation code ([`704eb00`](https://github.com/adamkarvonen/SAEBench/commit/704eb00a35fc4bdd66316cb25834722a13eeb77b))

* adding in k-sparse probing experiment code ([`0ab6d2c`](https://github.com/adamkarvonen/SAEBench/commit/0ab6d2c45f9588861dfec31f3c4104a2ac06279d))

* WIP: absorption ([`6348fb2`](https://github.com/adamkarvonen/SAEBench/commit/6348fb2fefdb60e360b03d5e26d391b5afbe1ea2))

* Update mdl_eval ([`bdefc02`](https://github.com/adamkarvonen/SAEBench/commit/bdefc022abf3ba4ce05ddd1624f078b4748fb371))

* Utils renaming ([`cbd6b99`](https://github.com/adamkarvonen/SAEBench/commit/cbd6b998d4da7767de956bfeff5e62881721e853))

* Scaffold mdl eval (orange) ([`a6b3406`](https://github.com/adamkarvonen/SAEBench/commit/a6b340690c3c89463ae658041c8148048323bd61))

* Merge pull request #4 from adamkarvonen/shift_eval

Shift eval ([`22a2a72`](https://github.com/adamkarvonen/SAEBench/commit/22a2a72e5e64372c6b91334722561b8c054934d2))

* Update README with determinism ([`aed96e8`](https://github.com/adamkarvonen/SAEBench/commit/aed96e84987b2d986ed713a1debc141038c910da))

* Fixed shift and tpp end to end tests ([`2c2947a`](https://github.com/adamkarvonen/SAEBench/commit/2c2947ab03af1e0def3de94405b7218b055037d9))

* Add README for shift and tpp ([`bcf3934`](https://github.com/adamkarvonen/SAEBench/commit/bcf3934c2b3f9455b5a1c80b4ce02dd83000d697))

* Add end to end test for shift and tpp ([`4ac13ef`](https://github.com/adamkarvonen/SAEBench/commit/4ac13ef7265aaefd000c3eb2b3752a4156decd26))

* Move SAE to model dtype, add option to set column1_vals_list ([`6ae4065`](https://github.com/adamkarvonen/SAEBench/commit/6ae4065f977eb78bb59ab014818fa85caec6dfbe))

* Initial working SHIFT / TPP evals ([`fab86d4`](https://github.com/adamkarvonen/SAEBench/commit/fab86d4b08a6c563f88d0ae4238da9c76d31b95f))

* Add SHIFT paired classes ([`900a04b`](https://github.com/adamkarvonen/SAEBench/commit/900a04b5dc50de980a81a1ce6f906f218f509fbb))

* Modify probe training for usage with SHIFT / TPP ([`039fd29`](https://github.com/adamkarvonen/SAEBench/commit/039fd29e2965f74424e883df4b185684881b7476))

* Pin dataset name for sparse probing test ([`26d85ed`](https://github.com/adamkarvonen/SAEBench/commit/26d85ede28a2307b95505cb397a377e4c5cc3f2b))

* Correct shape annotation ([`dc1f3d7`](https://github.com/adamkarvonen/SAEBench/commit/dc1f3d72f18eb8671acd5b5e2f3a92bec4077572))

* Merge pull request #2 from adamkarvonen/sparse_probing_add_datasets

Sparse probing add datasets ([`19b4c4a`](https://github.com/adamkarvonen/SAEBench/commit/19b4c4a8b7e2e4337d01353f3d31e259dde76e13))

* Check for columns with missing second group ([`1895318`](https://github.com/adamkarvonen/SAEBench/commit/18953186f91eacf4a6c78c4d9b8c9cbbfd6bab67))

* Run sparse probing eval on multiple datasets and average results ([`6e158b3`](https://github.com/adamkarvonen/SAEBench/commit/6e158b3a337f99f66b1d6508c89d2daad8471a80))

* Add function to average results from multiple runs ([`47af366`](https://github.com/adamkarvonen/SAEBench/commit/47af3662baee348f7d70afd99819dae5a2663768))

* Remove html files ([`9c42b2f`](https://github.com/adamkarvonen/SAEBench/commit/9c42b2f0f98efd5d9d5bc49acb312e8bd1525d68))

* Update READMEs ([`e6f1e3b`](https://github.com/adamkarvonen/SAEBench/commit/e6f1e3b94c10bcd90f9e3f0908762dda2a174f9f))

* Create end to end test for sparse probing repo ([`1eb63e9`](https://github.com/adamkarvonen/SAEBench/commit/1eb63e9f3db3a0f1bc2095556172cc5a4195d66c))

* Rename file so it isn&#39;t ran as a test ([`127924a`](https://github.com/adamkarvonen/SAEBench/commit/127924ac84c56e80bb593bb8d561396bde735f8e))

* Fix main.py imports, set seeds inside function ([`9ec51f2`](https://github.com/adamkarvonen/SAEBench/commit/9ec51f25346b3d1de6b3cdc827cb5aa2454cb1f5))

* Deprecate temporary fix for new sae_lens version ([`13e5da0`](https://github.com/adamkarvonen/SAEBench/commit/13e5da01bc0e26fa696146b4149e9306efdacfe0))

* Don&#39;t pin to specific versions ([`146c1fc`](https://github.com/adamkarvonen/SAEBench/commit/146c1fc545c922d3872bba632fc12362adda5076))

* Remove old requirements.txt ([`2ea8bac`](https://github.com/adamkarvonen/SAEBench/commit/2ea8baca42afae19be3e1ac9c3b8981f74c2f288))

* Merge pull request #1 from adamkarvonen/restructure

Restructure ([`38e2721`](https://github.com/adamkarvonen/SAEBench/commit/38e27213be6e67bd6d905686963d0b13be5f9c69))

* added
restructure ([`c3952bd`](https://github.com/adamkarvonen/SAEBench/commit/c3952bd231bc0fd4c1133d9d6f78321b3ded8e83))

* restructure ([`f328f7f`](https://github.com/adamkarvonen/SAEBench/commit/f328f7fc26fb308f564b11e195d534e8e333e941))

* created branch ([`374561a`](https://github.com/adamkarvonen/SAEBench/commit/374561a102d68c5f4620b82ff8e403e6645286c2))

* Update to use new eval results folder ([`69657d8`](https://github.com/adamkarvonen/SAEBench/commit/69657d8763edb978db8af448c23551e6c974aa08))

* Make selected SAEs input explicit ([`fdc0afd`](https://github.com/adamkarvonen/SAEBench/commit/fdc0afd49782e63be5d4c5f2a6b85e0b2d0657d7))

* mention SAE Lens tutorial ([`0121f32`](https://github.com/adamkarvonen/SAEBench/commit/0121f32546d576554899f7e2f48c058ce141a60f))

* Add pythia example results ([`e0a14be`](https://github.com/adamkarvonen/SAEBench/commit/e0a14be00d3503c6a9d26312aa6cea33677ec434))

* Fix titles ([`b7203b0`](https://github.com/adamkarvonen/SAEBench/commit/b7203b01f9c513f1ea998a61fe1a7589f06539f0))

* Demonstrate loading of Pythia and Gemma SAEs ([`d158304`](https://github.com/adamkarvonen/SAEBench/commit/d15830414750f35f1ce1485c8a21425365f177f0))

* Add new examples Gemma results ([`cc1a27c`](https://github.com/adamkarvonen/SAEBench/commit/cc1a27c8f37c2ed5465272f63830db2eab1fb442))

* Include checkpoints and not checkpoints if include_checkpoints is True ([`f5b1a71`](https://github.com/adamkarvonen/SAEBench/commit/f5b1a71a14730d5690b60bb6ef285def0a964451))

* Temp fix for SAE Bench TopK SAEs ([`8b7a6ec`](https://github.com/adamkarvonen/SAEBench/commit/8b7a6ec3c27bbde9452999274e8592a962f2bca9))

* Use sklearn by default, except for training on all SAE latents ([`2b1e2b6`](https://github.com/adamkarvonen/SAEBench/commit/2b1e2b6ee45c226ebf9e57a0f38ea468731f710f))

* Test file for experimenting with probe training ([`4cd9cff`](https://github.com/adamkarvonen/SAEBench/commit/4cd9cff2a702f9ec69909c1251af281992c86596))

* Add gemma-scope eval_results data ([`5f2bf51`](https://github.com/adamkarvonen/SAEBench/commit/5f2bf5154020b6b44b5c30b9d456ea483c751826))

* Use a dict of sae_release: list[sae_names] for the sparse probing eval ([`70100d8`](https://github.com/adamkarvonen/SAEBench/commit/70100d829e0d543e7fbcb3c95726956824319a03))

* Define llm dtype in activation_collection.py ([`0f29194`](https://github.com/adamkarvonen/SAEBench/commit/0f29194384c2fd2c34c5287a62a8f0ed4b9f5dc4))

* training plot scaled by steps ([`9b49ec4`](https://github.com/adamkarvonen/SAEBench/commit/9b49ec48a51e979715a283f44c0e30144afe172d))

* bugfix sae_bench prefix ([`da9f95f`](https://github.com/adamkarvonen/SAEBench/commit/da9f95f95112334abc2f7fcf80900e33de15f9fc))

* Merge branch &#39;main&#39; of https://github.com/adamkarvonen/SAE_Bench_Template into main ([`54a6156`](https://github.com/adamkarvonen/SAEBench/commit/54a61568bb16d601e9e8a7bb32bdc8884bbf31b9))

* Also calculate k-sparse probing for LLM activations ([`5e816b0`](https://github.com/adamkarvonen/SAEBench/commit/5e816b00bdf5a935fa251d219325119652a01c67))

* Move default results into sparse_probing/ from sparse_probing/src/ ([`2283cc5`](https://github.com/adamkarvonen/SAEBench/commit/2283cc54035f672540deec8a17d12b28a9bdfe46))

* By default, perform k-sparse probing using sklearn instead of hand-rolled implementation ([`57c4216`](https://github.com/adamkarvonen/SAEBench/commit/57c4216a82bf30ff52a872692b714ac6f3269b0c))

* Optional L1 penalty ([`078fb32`](https://github.com/adamkarvonen/SAEBench/commit/078fb32cc92afab3047c8bfdc3900178427fd6d5))

* Separate logic for selecting topk features using mean diff ([`8700aa7`](https://github.com/adamkarvonen/SAEBench/commit/8700aa7b59af68c8ef490939af505e78e302ddf6))

* Set dtype based on model ([`baa004e`](https://github.com/adamkarvonen/SAEBench/commit/baa004ef5d3291f88d928442b0f626cfcc3f8f53))

* Add type annotations so config parameters are saved ([`337c21d`](https://github.com/adamkarvonen/SAEBench/commit/337c21da9897c3064bdb0107ed3c8f0c010e289b))

* add plot over training steps ([`065825e`](https://github.com/adamkarvonen/SAEBench/commit/065825e6a3e64d3e4d78fd8c86239f65df5d3fef))

* add virtualenv instructions ([`68edafe`](https://github.com/adamkarvonen/SAEBench/commit/68edafe64c08f0cfaaacd13a6f099065d21f0908))

* added interactive 3var plot and renamed files for clarity ([`5afafb7`](https://github.com/adamkarvonen/SAEBench/commit/5afafb7a4f9dbdc41256854447de0a69f0fa3323))

* adapted requirements ([`aeadbb4`](https://github.com/adamkarvonen/SAEBench/commit/aeadbb47ee843f94c3b9430f4d0ead6c87200358))

* Merge branch &#39;main&#39; of https://github.com/adamkarvonen/SAE_Bench_Template into main ([`5ee11cf`](https://github.com/adamkarvonen/SAEBench/commit/5ee11cf1000d8ef755b43a7646b5d943a142d7aa))

* Update README.md ([`98d5f57`](https://github.com/adamkarvonen/SAEBench/commit/98d5f57a03dffaa481657b9154cb7197d2adfaeb))

* Update README.md ([`2e2e9f8`](https://github.com/adamkarvonen/SAEBench/commit/2e2e9f8a9197f8d121208b110c9d6c6e0157bae7))

* clarified README.md ([`205b2fb`](https://github.com/adamkarvonen/SAEBench/commit/205b2fbc9edbe604625be573cdf5e27e7278ee3a))

* debugging correlation plots ([`cee970f`](https://github.com/adamkarvonen/SAEBench/commit/cee970f7497ca92605d6814e0e916e47e0800bf4))

* moved formatting utils to external file ([`8fcc5da`](https://github.com/adamkarvonen/SAEBench/commit/8fcc5da903333a0d116c1ea9dc1007a57b72ce60))

* clarify README.md ([`540123b`](https://github.com/adamkarvonen/SAEBench/commit/540123b0b66075e70baae7f1d49d3b3c586e0963))

* added explanation to template ([`9df0fcb`](https://github.com/adamkarvonen/SAEBench/commit/9df0fcb9682804a739fc3300b585b04af758f2c9))

* Improve README ([`68927c9`](https://github.com/adamkarvonen/SAEBench/commit/68927c9dde3187e385adc96e7c9c5d6ca3071889))

* Updated pythia and gemma results ([`57ecbea`](https://github.com/adamkarvonen/SAEBench/commit/57ecbea927d7743c00a2ea911b1d8fbb8e4a1222))

* Improve graphing notebook ([`66d0b66`](https://github.com/adamkarvonen/SAEBench/commit/66d0b666c04a63e74fc47307494816fb2e2c1149))

* Apply nbstripout ([`d955146`](https://github.com/adamkarvonen/SAEBench/commit/d95514663948553496946f0953188de873a3e815))

* Walkthrough notebook of dictionary format ([`5246390`](https://github.com/adamkarvonen/SAEBench/commit/5246390afc93f3a624df75e59cdb8bf49aa05cdd))

* Add to .gitignore ([`c0b1b84`](https://github.com/adamkarvonen/SAEBench/commit/c0b1b845ec2343120ea7d9c6c321b2815672ce01))

* Utility notebook to compare multiple run results ([`27352d3`](https://github.com/adamkarvonen/SAEBench/commit/27352d3807d25ea87f1b8092d5d71eedc937e822))

* Improve SAE naming, use Gemma by default ([`c03e540`](https://github.com/adamkarvonen/SAEBench/commit/c03e5405900387376f5518541e1aa489a6b4d08f))

* Add READMEs ([`b1ea1b3`](https://github.com/adamkarvonen/SAEBench/commit/b1ea1b3297597d7d00ab3ac4d8cc3964c61e0318))

* Make determinstic, improve sae key naming ([`6a86bdb`](https://github.com/adamkarvonen/SAEBench/commit/6a86bdb72fb99cdccf69617d4fba3b5ec184d67e))

* Make sure to type check shapes ([`e67a8bc`](https://github.com/adamkarvonen/SAEBench/commit/e67a8bcba42273269d96991544e734d917c3f194))

* Archive development notebooks ([`7ecd360`](https://github.com/adamkarvonen/SAEBench/commit/7ecd360f813d4c477adddb02b65b6eb615e67910))

* Fix the recording name of saes ([`08ceced`](https://github.com/adamkarvonen/SAEBench/commit/08ceced251caf2221b219a0217066c21e3108f02))

* Add missing batch indexing ([`aeda80e`](https://github.com/adamkarvonen/SAEBench/commit/aeda80efbbd85392ab4b45435c1654ccc0aa8de2))

* Refactor batch processing to handle all sae_batch_size scenarios efficiently ([`c15d61d`](https://github.com/adamkarvonen/SAEBench/commit/c15d61db6034e748adc57dacdd304f367c6918e4))

* Fix reduced precision warning ([`36c9a8b`](https://github.com/adamkarvonen/SAEBench/commit/36c9a8b703c870d325d0b01756dbea5454c9eae0))

* correctly save results ([`f5dfabb`](https://github.com/adamkarvonen/SAEBench/commit/f5dfabb533ddfad2dc16536a3b92b6c1db529358))

* example results ([`8b1ec0c`](https://github.com/adamkarvonen/SAEBench/commit/8b1ec0ca70b31e169b606ea2aa9b93496f8762c1))

* Data on existing SAEs ([`7761e2b`](https://github.com/adamkarvonen/SAEBench/commit/7761e2ba73f826b6592ce296db6ba6c1171836c7))

* Cleanup ([`0f631d7`](https://github.com/adamkarvonen/SAEBench/commit/0f631d7c80f451cbd5997edaf0b6e4b4c8e23afa))

* Dev notebook ([`1412347`](https://github.com/adamkarvonen/SAEBench/commit/141234786ff8a4ad1daad29b28c502fa3bf632fc))

* sparse probing eval ([`fc789e5`](https://github.com/adamkarvonen/SAEBench/commit/fc789e56b03c0db20d191a1fa1c0ed01a99302e5))

* Create bias in bios dataset ([`0ff4cf1`](https://github.com/adamkarvonen/SAEBench/commit/0ff4cf1d05ca5a0fe3463771497106a765c6ca9f))

* Apply nbstripout ([`b993ecc`](https://github.com/adamkarvonen/SAEBench/commit/b993ecc782b03ce7eb2a8715b57a2deb846a7699))

* Beginning of plotting notebook ([`141fbd2`](https://github.com/adamkarvonen/SAEBench/commit/141fbd26a7c29b54f022001d2e06191b7cbb827d))

* initial commit ([`319300f`](https://github.com/adamkarvonen/SAEBench/commit/319300f0cf7a731efaa768a6ff32bff62bcd60c1))
