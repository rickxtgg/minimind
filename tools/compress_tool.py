#!/usr/bin/env python3

# ----------------------------------------------------------------------
# 文件名: compress_tool.py
# 作者: (rickxt)
# 日期: 2025-3-17
# 版本: 2.4
#
# 描述:
#   此脚本用于压缩或解压缩文件/目录。支持多种压缩格式、递归/非递归压缩、
#   排除文件、仅压缩目录下的文件、分别压缩子目录，并提供解压缩时的
#   覆盖/跳过/询问/重命名选项。同时支持命令行参数和交互式使用。
#   支持批量解压缩目录中的所有归档文件。
#
# 使用方法:
#   1. 命令行模式:
#      python compress_tool.py compress [-h] [-f FORMAT] [-r] [-e EXCLUDE [EXCLUDE ...]] [-m {dir,file,all}] target
#      python compress_tool.py decompress [-h] [-o OUTPUT] [-ov {yes,no,ask,rename}] archive [archive ...]
#
#   2. 交互式模式:
#      直接运行脚本: python compress_tool.py
#      按照提示进行操作。
#
# 注意事项:
#   - 如果存在同名的压缩文件，压缩时会覆盖现有文件。
#
# 依赖:
#   - Python 3 (建议 3.6 或更高版本)
#   - tqdm (可选，用于显示进度条)
# ----------------------------------------------------------------------

import os
import shutil
import sys
import argparse
import zipfile
import tarfile
import fnmatch
import glob  # 导入 glob 模块
from tqdm import tqdm

def exclude_filter(tarinfo, exclude_patterns):
    """用于 tarfile.add 的 filter 函数, 排除文件"""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(tarinfo.name, pattern):
            return None
    return tarinfo

def compress_directory(directory, format_choice, recursive=False, exclude=None, mode='all'):
    """压缩目录 (只保留zip格式)"""
    if format_choice == '1':
        archive_format = 'zip'
        extension = '.zip'
    else:
        print("无效的压缩格式。只支持zip格式。")
        return

    if exclude is None:
        exclude = []

    try:
        if recursive:  # 将整个目录压缩为一个文件
            base_name = os.path.basename(directory)
            archive_name = base_name + extension
            # 创建完整的压缩文件路径
            archive_path = os.path.join(os.getcwd(), archive_name)
            
            # 将压缩文件本身添加到排除列表中，防止递归压缩
            exclude.append(archive_path)
            
            with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                with tqdm(total=sum(1 for root, _, files in os.walk(directory) for file in files), unit="file", desc=f"正在压缩 {base_name}") as pbar:
                    for root, _, files in os.walk(directory):
                        for file in files:
                            file_path = os.path.join(root, file)
                            # 检查是否是压缩文件本身或匹配排除模式
                            if file_path == archive_path or any(fnmatch.fnmatch(file_path, pattern) for pattern in exclude):
                                continue
                            zf.write(file_path, arcname=os.path.relpath(file_path, directory))
                            pbar.update(1)

            print(f"已创建: {archive_name}")

        else: # 非递归压缩
            for item in os.listdir(directory):
                item_path = os.path.join(directory,item)
                if any(fnmatch.fnmatch(item_path, pattern) for pattern in exclude):
                    continue

                if mode == 'all' or (mode == 'dir' and os.path.isdir(item_path)) or (mode == 'file' and os.path.isfile(item_path)):
                    if os.path.isdir(item_path):
                        base_name = os.path.basename(item_path)
                        archive_name = base_name + extension

                        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                            with tqdm(total=len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path,f))]), unit="file", desc=f"正在压缩 {base_name}") as pbar:  # 仅统计文件数量
                                for file in os.listdir(item_path):  # 只压缩当前层级
                                    file_path = os.path.join(item_path, file)
                                    if os.path.isfile(file_path):
                                        zf.write(file_path, arcname=file)
                                        pbar.update(1)

                    elif os.path.isfile(item_path):
                        base_name = os.path.splitext(item)[0]
                        archive_name = base_name + extension
                        with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as zf:
                            with tqdm(total=1, unit="file", desc=f"正在压缩 {base_name}") as pbar:
                                zf.write(item_path, arcname=item)
                                pbar.update(1)

    except shutil.Error as e:
        print(f"压缩失败: {e}")
    except OSError as e:
        print(f"压缩失败: 操作系统错误: {e}")
    except ValueError as e:
        print(f"压缩失败: 参数错误: {e}")
    except Exception as e:
        print(f"压缩失败: 未知错误: {e}")



def decompress_archive(archive_path, extract_dir=None, overwrite='ask'):
    """解压缩单个归档文件 (与之前版本基本相同，但增加了对 extract_dir 不存在时的处理)"""

    if extract_dir is None:
        base_name = os.path.splitext(os.path.basename(archive_path))[0]
        while "." in base_name:
            base_name, ext = os.path.splitext(base_name)
            if ext == ".tar":
                break
        extract_dir = os.path.join(os.path.dirname(archive_path), base_name)

    # 确保输出目录存在
    if not os.path.exists(extract_dir):
        try:
            os.makedirs(extract_dir)  # 创建目录 (包括父目录)
            print(f"已创建输出目录: {extract_dir}")
        except OSError as e:
            print(f"创建输出目录失败: {extract_dir} - {e}")
            return  # 如果创建目录失败，则退出函数

    try:
        if archive_path.endswith(".zip"):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                with tqdm(total=len(zf.infolist()), unit="file", desc=f"解压 {os.path.basename(archive_path)}") as pbar:
                    for member in zf.infolist():
                        target_path = os.path.join(extract_dir, member.filename)

                        if os.path.exists(target_path):
                            if overwrite == 'yes':
                                if os.path.isfile(target_path):
                                    os.remove(target_path)
                                else:
                                    shutil.rmtree(target_path)
                            elif overwrite == 'no':
                                pbar.update(1)  # 即使跳过，也要更新进度条
                                continue
                            elif overwrite == 'rename':
                                # 重命名逻辑
                                counter = 1
                                while True:
                                    base, ext = os.path.splitext(target_path)
                                    new_path = f"{base}_{counter}{ext}"
                                    if not os.path.exists(new_path):
                                        target_path = new_path
                                        break
                                    counter += 1
                            elif overwrite == 'ask':
                                while True:
                                    response = input(f"文件/目录 '{target_path}' 已存在。 覆盖(y)/跳过(n)/全部覆盖(a)/全部跳过(q)/重命名(r)?: [y/n/a/q/r] ").lower()
                                    if response in ('y', 'n', 'a', 'q', 'r'):
                                        break
                                    print("无效的输入。")

                                if response == 'n':
                                    pbar.update(1)
                                    continue
                                elif response == 'a':
                                    overwrite = 'yes'
                                    if os.path.isfile(target_path):
                                        os.remove(target_path)
                                    else:
                                        shutil.rmtree(target_path)
                                elif response == 'q':
                                    return
                                elif response == 'r':
                                    # 重命名
                                    counter = 1
                                    while True:
                                      base, ext = os.path.splitext(target_path)
                                      new_path = f"{base}_{counter}{ext}"
                                      if not os.path.exists(new_path):
                                        target_path = new_path
                                        break
                                      counter+=1

                        zf.extract(member, path=extract_dir)  # 解压
                        if target_path != os.path.join(extract_dir, member.filename):
                            # 如果发生了重命名，则移动文件
                            os.rename(os.path.join(extract_dir, member.filename), target_path)
                        pbar.update(1)

        elif any(archive_path.endswith(ext) for ext in (".tar", ".tar.gz", ".tar.bz2", ".tar.xz")):
            with tarfile.open(archive_path, 'r:*') as tar:
                with tqdm(total=len(tar.getmembers()), unit="file", desc=f"解压 {os.path.basename(archive_path)}") as pbar:
                    for member in tar.getmembers():
                        target_path = os.path.join(extract_dir, member.name)

                        if os.path.exists(target_path):
                            if overwrite == 'yes':
                                if member.isfile():
                                    os.remove(target_path)
                                elif member.isdir():
                                    shutil.rmtree(target_path)
                            elif overwrite == 'no':
                                pbar.update(1)
                                continue
                            elif overwrite == 'rename':
                                counter = 1
                                while True:
                                     base, ext = os.path.splitext(target_path)
                                     new_path = f"{base}_{counter}{ext if member.isfile() else ''}"  #目录不加原扩展名
                                     if not os.path.exists(new_path):
                                        target_path = new_path
                                        break
                                     counter += 1

                            elif overwrite == 'ask':
                                while True:
                                    response = input(f"文件/目录 '{target_path}' 已存在。 覆盖(y)/跳过(n)/全部覆盖(a)/全部跳过(q)/重命名(r)?: [y/n/a/q/r] ").lower()
                                    if response in ('y', 'n', 'a', 'q', 'r'):
                                        break
                                    print("无效的输入。")

                                if response == 'n':
                                    pbar.update(1)
                                    continue
                                elif response == 'a':
                                    overwrite = 'yes'
                                    if member.isfile():
                                        os.remove(target_path)
                                    elif member.isdir():
                                        shutil.rmtree(target_path)
                                elif response == 'q':
                                    return
                                elif response == 'r':
                                    counter = 1
                                    while True:
                                        base, ext = os.path.splitext(target_path)
                                        new_path = f"{base}_{counter}{ext if member.isfile() else ''}" #目录不加扩展名
                                        if not os.path.exists(new_path):
                                          target_path = new_path
                                          break
                                        counter +=1

                        tar.extract(member, path=extract_dir) #解压

                        if target_path != os.path.join(extract_dir, member.name) : #如果发生了重命名，则移动文件
                            # 重命名文件或目录.  member.name 不一定是文件, 可能是目录
                            if member.isdir():
                                # 如果是目录，需要递归移动目录内容
                                shutil.move(os.path.join(extract_dir, member.name), target_path)
                            else: #文件
                                os.rename(os.path.join(extract_dir, member.name), target_path)


                        pbar.update(1)
        else:
            print("不支持的文件类型")
            return

        print(f"已解压缩: {archive_path} 到 {extract_dir}")

    except FileNotFoundError:
        print(f"错误：文件 '{archive_path}' 不存在。")
    except (zipfile.BadZipFile, tarfile.ReadError) as e:
        print(f"错误：文件 '{archive_path}' 损坏或不是有效的归档文件: {e}")
    except OSError as e:
        print(f"错误：操作系统错误: {e} (可能权限不足，或者磁盘已满)")
    except Exception as e:
        print(f"解压缩失败: {archive_path} - {e}")



def decompress_multiple_archives(archives, output_dir, overwrite='ask'):
    """解压缩多个归档文件 (或目录中的所有归档文件)"""
    if not archives:
        print("没有要解压缩的文件。")
        return

    # 修改这部分逻辑，不再使用第一个归档文件的目录作为默认输出目录
    # 而是在处理每个归档文件时单独确定其输出目录
    if output_dir is not None and not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"已创建输出目录：{output_dir}")
        except:
            print(f"创建输出目录失败：{output_dir}")
            return

    for archive_path in archives:
        if os.path.isfile(archive_path):
            # 为每个压缩文件创建单独的解压目录
            file_basename = os.path.splitext(os.path.basename(archive_path))[0]
            # 处理 .tar.gz 等多重扩展名
            while "." in file_basename:
                name, ext = os.path.splitext(file_basename)
                if ext == ".tar":
                    break
                file_basename = name
            
            # 如果指定了输出目录，则在输出目录下创建与压缩文件同名的子目录
            # 否则，使用压缩文件所在目录
            if output_dir is not None:
                extract_subdir = os.path.join(output_dir, file_basename)
            else:
                extract_subdir = os.path.join(os.path.dirname(archive_path), file_basename)
                
            decompress_archive(archive_path, extract_subdir, overwrite)  # 解压缩单个文件
        elif os.path.isdir(archive_path):
            # 如果是目录，则遍历目录中的所有归档文件
            for root, _, files in os.walk(archive_path):  # 递归遍历
                for file in files:
                    file_path = os.path.join(root, file)
                    # 检查文件扩展名是否受支持
                    if any(file_path.endswith(ext) for ext in (".zip", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz")):
                        # 为每个压缩文件创建单独的解压目录
                        file_basename = os.path.splitext(os.path.basename(file_path))[0]
                        # 处理 .tar.gz 等多重扩展名
                        while "." in file_basename:
                            name, ext = os.path.splitext(file_basename)
                            if ext == ".tar":
                                break
                            file_basename = name
                        
                        # 如果指定了输出目录，则在输出目录下创建与压缩文件同名的子目录
                        # 否则，使用压缩文件所在目录
                        if output_dir is not None:
                            extract_subdir = os.path.join(output_dir, file_basename)
                        else:
                            extract_subdir = os.path.join(os.path.dirname(file_path), file_basename)
                            
                        decompress_archive(file_path, extract_subdir, overwrite)
        else: # 通配符
            expanded_paths = glob.glob(archive_path) #展开
            if not expanded_paths:
                print(f"错误：找不到匹配 '{archive_path}' 的文件/目录。")
                continue # 没有匹配项

            for path in expanded_paths:
                if os.path.isfile(path):
                    # 为每个压缩文件创建单独的解压目录
                    file_basename = os.path.splitext(os.path.basename(path))[0]
                    # 处理 .tar.gz 等多重扩展名
                    while "." in file_basename:
                        name, ext = os.path.splitext(file_basename)
                        if ext == ".tar":
                            break
                        file_basename = name
                    
                    # 如果指定了输出目录，则在输出目录下创建与压缩文件同名的子目录
                    # 否则，使用压缩文件所在目录
                    if output_dir is not None:
                        extract_subdir = os.path.join(output_dir, file_basename)
                    else:
                        extract_subdir = os.path.join(os.path.dirname(path), file_basename)
                        
                    decompress_archive(path, extract_subdir, overwrite)
                elif os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            if any(file_path.endswith(ext) for ext in (".zip", ".tar", ".tar.gz", ".tar.bz2", ".tar.xz")):
                                # 为每个压缩文件创建单独的解压目录
                                file_basename = os.path.splitext(os.path.basename(file_path))[0]
                                # 处理 .tar.gz 等多重扩展名
                                while "." in file_basename:
                                    name, ext = os.path.splitext(file_basename)
                                    if ext == ".tar":
                                        break
                                    file_basename = name
                                
                                # 如果指定了输出目录，则在输出目录下创建与压缩文件同名的子目录
                                # 否则，使用压缩文件所在目录
                                if output_dir is not None:
                                    extract_subdir = os.path.join(output_dir, file_basename)
                                else:
                                    extract_subdir = os.path.join(os.path.dirname(file_path), file_basename)
                                    
                                decompress_archive(file_path, extract_subdir, overwrite)

def interactive_mode():
    """交互模式"""
    while True:
        print("\n请选择操作：")
        print("1. 压缩")
        print("2. 解压缩")
        print("3. 退出")

        choice = input("请输入选项 (1-3): ")

        if choice == '3':
            print("退出。")
            break

        if choice == '1':  # 压缩
            target = input("请输入要压缩的目录或文件路径 (留空表示当前目录): ")
            target = os.path.abspath(target) if target else os.getcwd()
            if not os.path.exists(target):
                print("错误：目标路径不存在。")
                continue

            print("压缩格式：zip")
            format_choice = '1'  # 只支持zip格式

            print("请选择压缩方式：")
            print("1. 将整个目录压缩为一个文件")
            print("2. 将目录下的每个子目录分别压缩")
            print("3. 将目录下的每个文件分别压缩")
            print("4. 将目录下的每个子目录和文件分别压缩")
            recursive_choice = input("请输入选项 (1-4): ")

            if recursive_choice == '1':
                recursive = True
                mode = 'all'  # 不重要，因为 recursive=True
            elif recursive_choice in ('2', '3', '4'):
                recursive = False
                if recursive_choice == '2':
                    mode = 'dir'
                elif recursive_choice == '3':
                    mode = 'file'
                else:
                    mode = 'all'
            else:
                print("错误：无效的压缩方式。")
                continue

            exclude_str = input("请输入要排除的文件/目录，以空格分隔 (支持通配符，留空表示不排除): ")
            exclude = exclude_str.split() if exclude_str else None

            compress_directory(target, format_choice, recursive, exclude, mode)

        elif choice == '2':  # 解压缩
            archive_input = input("请输入要解压缩的归档文件路径、目录或通配符 (多个以空格分隔，留空表示当前目录): ")
            archives = archive_input.split() if archive_input else ['.']  # 默认为当前目录
            archives = [os.path.abspath(a) for a in archives]  # 转换为绝对路径


            output = input("请输入输出目录 (留空表示与归档文件同名目录): ")
            output = os.path.abspath(output) if output else None

            print("请选择解压缩选项：")
            print("1. 覆盖已存在的文件/目录")
            print("2. 跳过已存在的文件/目录")
            print("3. 询问每个已存在的文件/目录")
            print("4. 重命名已存在的文件/目录")
            overwrite_choice = input("请输入选项 (1-4): ")
            if overwrite_choice == '1':
                overwrite = 'yes'
            elif overwrite_choice == '2':
                overwrite = 'no'
            elif overwrite_choice == '3':
                overwrite = 'ask'
            elif overwrite_choice == '4':
                overwrite = 'rename'
            else:
                print("错误：无效的解压缩选项。")
                continue

            decompress_multiple_archives(archives, output, overwrite)

        else:
            print("无效选项")

def main():
    if len(sys.argv) > 1:
        # 命令行模式的 argparse 部分
        parser = argparse.ArgumentParser(description="压缩或解压缩文件/目录")
        subparsers = parser.add_subparsers(dest='command', help='操作 (compress/decompress)')

        # 压缩子命令
        compress_parser = subparsers.add_parser('compress', help='压缩文件/目录')
        compress_parser.add_argument('target', help='要压缩的目录或文件')
        compress_parser.add_argument('-f', '--format', choices=['zip'], default='zip', help='压缩格式 (仅支持zip)')
        compress_parser.add_argument('-r', '--recursive', action='store_true', help='递归压缩,将整个目录及其内容压缩为一个文件')
        compress_parser.add_argument('-e', '--exclude', nargs='+', help='要排除的文件/目录 (支持通配符)')
        compress_parser.add_argument('-m', '--mode', choices=['dir', 'file', 'all'], default='all', help='非递归压缩模式 (默认: all)')

        # 解压缩子命令
        decompress_parser = subparsers.add_parser('decompress', help='解压缩文件')
        decompress_parser.add_argument('archive', nargs='+', help='要解压缩的归档文件、目录或通配符')  # 支持多个输入
        decompress_parser.add_argument('-o', '--output', help='输出目录 (默认与归档文件同名目录)')
        decompress_parser.add_argument('-ov', '--overwrite', choices=['yes', 'no', 'ask', 'rename'], default='ask', help='如果目标已存在，覆盖/跳过/询问/重命名 (默认: ask)')

        args = parser.parse_args()

        if args.command == 'compress':
            if not os.path.exists(args.target):
                print(f"错误： 目标 '{args.target}' 不存在.")
                sys.exit(1)

            if os.path.isfile(args.target):
                # 命令行模式下，如果目标是文件，强制 recursive=True, mode='all'
                compress_directory(args.target, '1', True, args.exclude, 'all')
            else:
                compress_directory(args.target, '1', args.recursive, args.exclude, args.mode)

        elif args.command == 'decompress':
            decompress_multiple_archives(args.archive, args.output, args.overwrite)

        else:
            parser.print_help()
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
