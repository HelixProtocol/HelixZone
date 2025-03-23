"""Cross-platform packaging configuration for HelixZone."""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import json

@dataclass
class PackageConfig:
    """Configuration for package building."""
    app_name: str = "HelixZone"
    version: str = "1.0.0"
    author: str = "HelixZone Team"
    description: str = "Advanced Image Processing Application"
    entry_point: str = "src.helixzone.main:main"
    icon_path: Optional[str] = None
    
    # Platform-specific options
    windows_options: Dict[str, str] = field(default_factory=dict)
    macos_options: Dict[str, str] = field(default_factory=dict)
    linux_options: Dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize platform-specific options."""
        self.windows_options = {
            'installer_name': f"{self.app_name}-{self.version}-setup.exe",
            'nsis_template': 'installer.nsi',
            'requires_admin': 'false',
            'install_dir': f'Program Files\\{self.app_name}'
        }
        
        self.macos_options = {
            'dmg_name': f"{self.app_name}-{self.version}.dmg",
            'app_category': 'public.app-category.graphics-design',
            'bundle_identifier': 'com.helixzone.app',
            'codesign_identity': 'none'
        }
        
        self.linux_options = {
            'appimage_name': f"{self.app_name}-{self.version}.AppImage",
            'desktop_entry': f"{self.app_name}.desktop",
            'categories': 'Graphics;2DGraphics;RasterGraphics;',
            'install_dir': f'/opt/{self.app_name.lower()}'
        }

def generate_pyinstaller_spec(config: PackageConfig) -> str:
    """Generate PyInstaller spec file content.
    
    Args:
        config: Package configuration
        
    Returns:
        Content of the spec file
    """
    return f'''# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{config.entry_point.replace(":", "/")}'],
    pathex=[],
    binaries=[],
    datas=[
        ('src/helixzone/resources', 'resources'),
        ('src/helixzone/gui/styles', 'styles')
    ],
    hiddenimports=[
        'numpy',
        'cv2',
        'PyQt6',
        'cupy',
        'pyopencl'
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
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
    name='{config.app_name}',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='{config.icon_path or "NONE"}',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='{config.app_name}',
)

# Platform-specific options
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='{config.app_name}.app',
        icon='{config.icon_path or "NONE"}',
        bundle_identifier='{config.macos_options["bundle_identifier"]}',
        info_plist={{
            'CFBundleShortVersionString': '{config.version}',
            'CFBundleVersion': '{config.version}',
            'CFBundleName': '{config.app_name}',
            'CFBundleDisplayName': '{config.app_name}',
            'CFBundleIdentifier': '{config.macos_options["bundle_identifier"]}',
            'CFBundlePackageType': 'APPL',
            'CFBundleSignature': '????',
            'LSApplicationCategoryType': '{config.macos_options["app_category"]}',
            'NSHighResolutionCapable': True,
        }},
    )
'''

def generate_nsis_script(config: PackageConfig) -> str:
    """Generate NSIS installer script for Windows.
    
    Args:
        config: Package configuration
        
    Returns:
        Content of the NSIS script
    """
    return f'''# NSIS Script for {config.app_name}
!include "MUI2.nsh"

Name "{config.app_name}"
OutFile "{config.windows_options['installer_name']}"
InstallDir "$PROGRAMFILES\\{config.app_name}"
RequestExecutionLevel {config.windows_options['requires_admin']}

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

Section "Install"
    SetOutPath $INSTDIR
    File /r "dist\\{config.app_name}\\*.*"
    
    # Create start menu shortcut
    CreateDirectory "$SMPROGRAMS\\{config.app_name}"
    CreateShortCut "$SMPROGRAMS\\{config.app_name}\\{config.app_name}.lnk" "$INSTDIR\\{config.app_name}.exe"
    
    # Create uninstaller
    WriteUninstaller "$INSTDIR\\Uninstall.exe"
    
    # Add uninstall information to Add/Remove Programs
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{config.app_name}" \\
                     "DisplayName" "{config.app_name}"
    WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{config.app_name}" \\
                     "UninstallString" "$INSTDIR\\Uninstall.exe"
SectionEnd

Section "Uninstall"
    # Remove program files
    RMDir /r "$INSTDIR"
    
    # Remove start menu items
    RMDir /r "$SMPROGRAMS\\{config.app_name}"
    
    # Remove uninstall information
    DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{config.app_name}"
SectionEnd
'''

def generate_desktop_entry(config: PackageConfig) -> str:
    """Generate Linux .desktop file.
    
    Args:
        config: Package configuration
        
    Returns:
        Content of the .desktop file
    """
    return f'''[Desktop Entry]
Version={config.version}
Name={config.app_name}
Comment={config.description}
Exec={config.linux_options['install_dir']}/{config.app_name.lower()}
Icon={config.app_name.lower()}
Terminal=false
Type=Application
Categories={config.linux_options['categories']}
'''

def save_packaging_files(config: PackageConfig, output_dir: str = "packaging"):
    """Save all packaging configuration files.
    
    Args:
        config: Package configuration
        output_dir: Directory to save files in
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save PyInstaller spec
    spec_path = output_path / f"{config.app_name.lower()}.spec"
    spec_path.write_text(generate_pyinstaller_spec(config))
    
    # Save NSIS script for Windows
    nsis_path = output_path / config.windows_options['nsis_template']
    nsis_path.write_text(generate_nsis_script(config))
    
    # Save desktop entry for Linux
    desktop_path = output_path / config.linux_options['desktop_entry']
    desktop_path.write_text(generate_desktop_entry(config))
    
    # Save configuration as JSON
    config_path = output_path / "package_config.json"
    config_dict = {
        k: v for k, v in config.__dict__.items()
        if not k.startswith('_')
    }
    config_path.write_text(json.dumps(config_dict, indent=4))

def main():
    """Create packaging configuration files."""
    config = PackageConfig()
    save_packaging_files(config)

if __name__ == '__main__':
    main() 