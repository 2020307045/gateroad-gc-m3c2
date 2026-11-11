# 上传 GitHub

当前目录已整理为仅代码的仓库交付包，尚未创建远程仓库或上传。

建议名称：`gateroad-gc-m3c2`

建议简介：`Curvature-constrained gateroad point-cloud denoising and two-epoch GC-M3C2 deformation analysis.`

## 网页上传

1. 在 GitHub 新建仓库，按作者意愿选择公开或私有。
2. 将本项目目录内的文件与 `scripts/`、`docs/`、`tests/` 文件夹上传。确保 `.gitignore` 和 `.gitattributes` 一并上传。
3. 不要只上传 ZIP 文件；GitHub 应直接显示 README 和源码。
4. 确定许可后添加 LICENSE，论文正式发表后补充 DOI 和引用信息。

## Git 命令上传

先在 GitHub 创建空仓库。然后在本项目目录打开终端，逐行执行：

```bash
git init
git add .
git status --short
git commit -m "Prepare manuscript code release"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/gateroad-gc-m3c2.git
git push -u origin main
```

将 `YOUR_USERNAME` 替换为实际账号。`git status` 应只显示源码、测试、文档和依赖配置。

## 论文代码声明

远程仓库确实发布并可访问后，再将真实链接写入论文。当前交付没有生成不存在的 GitHub 地址，也没有将未选定的许可证写入项目。
