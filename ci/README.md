# GitHub Actions 自动部署

正式流水线位于 `.github/workflows/ci.yml`，推送到 `dev` 后依次执行 verify、test、build-and-push、deploy。`master` 和 Pull Request 只验证，不更新正式网站；可以在 `dev` 上使用 workflow_dispatch 重跑。只在本地 commit 不会触发 GitHub，必须执行 `git push origin dev`。

正式网站为 https://kratos.qzz.io/，服务器为 138.68.21.119，Compose 项目位于 `/srv/se3`。构建发生在 GitHub runner，按提交 SHA 标记镜像，生产服务器拉取准确的 sha256 digest。前端测试包含生产镜像启动、JS/CSS 静态资源、SPA 路由、版本和缓存标头；后端执行全部 pytest 测试。部署后检查容器健康、数据库连通性和公网版本。前后端通过服务器 flock 串行部署，失败时自动恢复上一容器镜像；更新容器期间可能有短暂连接中断。

仓库已经配置 DEPLOY_HOST、DEPLOY_USER、DEPLOY_SSH_KEY、DEPLOY_KNOWN_HOSTS 四个 Actions secrets。SSH 密钥为专用密钥，服务端 authorized_keys 使用 restrict 和 forced command，禁止任意 shell、端口转发和 PTY，只允许该仓库部署自己的服务。GitHub 的 GITHUB_TOKEN 只用于临时登录 GHCR，部署结束删除服务器临时登录文件。服务器现有 `.env`、数据库、证书和上传卷保留，不需要 DEPLOY_ENV_FILE。

服务器管理员已安装 `/usr/local/sbin/kratos-deploy`，对应受版本控制的源文件为 `ci/server-deploy.sh`。修改该脚本后，管理员需要通过本机 `scp ci/server-deploy.sh do:/usr/local/sbin/kratos-deploy` 同步，并执行 `ssh do 'chmod 700 /usr/local/sbin/kratos-deploy'`，日常业务代码修改不需要这个操作。持久化镜像覆盖文件为 `/srv/se3/deploy/backend.yml` 与 `/srv/se3/deploy/frontend.yml`，部署记录位于同目录的 backend.json/frontend.json。

后端部署前自动执行 pg_dump，备份保存在 `/srv/se3/deploy/backups/<commit>.dump`。容器回滚不会自动撤销数据库 schema 变更；今后如果引入破坏性迁移，需要单独设计兼容迁移或由管理员从该备份恢复数据库。备份及回滚镜像默认保留，不自动删除，磁盘不足时由管理员确认需要保留的版本后清理。

正常发布只需在本地修改、提交并执行 `git push origin dev`，然后到仓库 Actions 查看结果。若需要恢复某次业务版本，应通过 `git revert <commit>` 并推送 dev 触发完整验证和部署，不要直接改生产目录。旧 `.gitlab-ci.yml` 和 deploy_backend.sh 只保留作历史参考，GitHub 不使用它们。
