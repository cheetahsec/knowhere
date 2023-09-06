# Licensed to the LF AI & Data foundation under one
# or more contributor license agreements. See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership. The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# 1. 设置脚本路径和根目录变量：
$ScriptPath = $MyInvocation.MyCommand.Path
$RootDir = Split-Path (Resolve-Path $ScriptPath) -Parent

# 2. 设置操作系统名称，默认为ubuntu20.04：
$env:OS_NAME = if ($env:OS_NAME) { $env:OS_NAME } else { "ubuntu20.04" }

# 3. 获取操作系统信息，并根据操作系统类型设置$machine变量：
$unameOut = (Get-CimInstance -ClassName Win32_OperatingSystem).Caption
switch -Wildcard ($unameOut) {
    "Microsoft Windows *" { $machine = "Windows" }
    default { $machine = "UNKNOWN:$unameOut" }
}

# 4. 获取当前用户的UID和GID：
$uid = [System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value
$firstGroupSID = (Get-LocalUser -SID $uid).Groups | Select-Object -First 1 -ExpandProperty SID
$gid = if ($firstGroupSID) { (Get-LocalGroup -SID $firstGroupSID).Name } else { "unknown" }
# 5. 检查脚本参数是否为build，如果是，则设置$CHECK_BUILDER为1：
if ($args[0] -eq "build") {
    $CHECK_BUILDER = 1
}

# 6. 根据$CHECK_BUILDER的值，修改docker-compose.yml文件并生成docker-compose-devcontainer.yml文件：
if ($CHECK_BUILDER -eq 1) {
    (Get-Content "$RootDir/docker-compose.yml" | ForEach-Object { if ($_ -match "# Command") { $c = 5 } if ($c -gt 0) { $_ -replace '^', '#' } else { $_ } $c-- }) | Set-Content "$RootDir/docker-compose-devcontainer.yml"
} else {
    (Get-Content "$RootDir/docker-compose.yml" | ForEach-Object { if ($_ -match "# Build devcontainer") { $c = 6 } if ($c -gt 0) { $_ -replace '^', '#' } else { $_ } $c-- }) | Set-Content "$RootDir/docker-compose-devcontainer.yml.tmp"
    (Get-Content "$RootDir/docker-compose-devcontainer.yml.tmp" | ForEach-Object { if ($_ -match "# Command") { $c = 5 } if ($c -gt 0) { $_ -replace '^', '#' } else { $_ } $c-- }) | Set-Content "$RootDir/docker-compose-devcontainer.yml"
    Remove-Item "$RootDir/docker-compose-devcontainer.yml.tmp"
}

# 7. 替换docker-compose-devcontainer.yml文件中的用户ID和组ID：
# (Get-Content "$RootDir/docker-compose-devcontainer.yml") -replace '# user: {{ CURRENT_ID }}', "user: `"${uid}:${gid}`"" | Set-Content "$RootDir/docker-compose-devcontainer.yml"

# 8. 切换到根目录，并创建所需的Docker卷目录：
Push-Location $RootDir
$DockerVolumeDir = "${env:DOCKER_VOLUME_DIRECTORY:-.docker}"
New-Item -ItemType Directory -Path "$DockerVolumeDir/amd64-${env:OS_NAME}-ccache" -Force | Out-Null
New-Item -ItemType Directory -Path "$DockerVolumeDir/amd64-${env:OS_NAME}-go-mod" -Force | Out-Null
New-Item -ItemType Directory -Path "$DockerVolumeDir/amd64-${env:OS_NAME}-vscode-extensions" -Force | Out-Null
New-Item -ItemType Directory -Path "$DockerVolumeDir/amd64-${env:OS_NAME}-conan" -Force | Out-Null
Get-ChildItem -Path $DockerVolumeDir -Recurse | ForEach-Object { $_.Attributes = 'Normal' }

# 9. 根据脚本参数执行相应的Docker Compose命令：
if ($args[0] -eq "build") {
    docker-compose -f "$RootDir/docker-compose-devcontainer.yml" pull --ignore-pull-failures builder
    docker-compose -f "$RootDir/docker-compose-devcontainer.yml" build builder
}

if ($args[0] -eq "up") {
    docker-compose -f "$RootDir/docker-compose-devcontainer.yml" up -d
}

if ($args[0] -eq "down") {
    docker-compose -f "$RootDir/docker-compose-devcontainer.yml" down
}
# 10. 返回到之前的目录：
Pop-Location