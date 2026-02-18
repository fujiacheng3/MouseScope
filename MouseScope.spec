# -*- mode: python ; coding: utf-8 -*-
"""
MouseScope Desktop — PyInstaller 打包配置
用法:  pyinstaller MouseScope.spec
"""

import os
import sys

block_cipher = None

# 项目根目录
PROJECT_DIR = os.path.dirname(os.path.abspath(SPEC))

a = Analysis(
    [os.path.join(PROJECT_DIR, 'app.py')],
    pathex=[PROJECT_DIR, os.path.join(PROJECT_DIR, 'core')],
    binaries=[],
    datas=[
        # 模板和静态资源
        (os.path.join(PROJECT_DIR, 'templates'), 'templates'),
        (os.path.join(PROJECT_DIR, 'static'), 'static'),
        # core 模块
        (os.path.join(PROJECT_DIR, 'core'), 'core'),
        # 许可证
        (os.path.join(PROJECT_DIR, 'LICENSE'), '.'),
    ],
    hiddenimports=[
        'flask',
        'jinja2',
        'markupsafe',
        'cv2',
        'numpy',
        'pandas',
        'engineio.async_drivers.threading',
        'vision_tracker',
        'performance_utils',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 排除不需要的大模块，减小体积
        'tkinter',
        'matplotlib',
        'scipy',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
        'ultralytics',  # YOLO 可选，按需取消注释
        'torch',
        'torchvision',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='MouseScope',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,       # 保留控制台窗口，方便看日志；发布时可改 False
    # icon='static/img/icon.ico',  # 如果有图标，取消注释
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='MouseScope',
)
