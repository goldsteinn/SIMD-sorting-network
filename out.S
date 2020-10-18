
driver:     file format elf64-x86-64


Disassembly of section .init:

0000000000001000 <_init>:
    1000:	f3 0f 1e fa          	endbr64 
    1004:	48 83 ec 08          	sub    $0x8,%rsp
    1008:	48 8b 05 d9 3f 00 00 	mov    0x3fd9(%rip),%rax        # 4fe8 <__gmon_start__>
    100f:	48 85 c0             	test   %rax,%rax
    1012:	74 02                	je     1016 <_init+0x16>
    1014:	ff d0                	callq  *%rax
    1016:	48 83 c4 08          	add    $0x8,%rsp
    101a:	c3                   	retq   

Disassembly of section .plt:

0000000000001020 <.plt>:
    1020:	ff 35 6a 3f 00 00    	pushq  0x3f6a(%rip)        # 4f90 <_GLOBAL_OFFSET_TABLE_+0x8>
    1026:	f2 ff 25 6b 3f 00 00 	bnd jmpq *0x3f6b(%rip)        # 4f98 <_GLOBAL_OFFSET_TABLE_+0x10>
    102d:	0f 1f 00             	nopl   (%rax)
    1030:	f3 0f 1e fa          	endbr64 
    1034:	68 00 00 00 00       	pushq  $0x0
    1039:	f2 e9 e1 ff ff ff    	bnd jmpq 1020 <.plt>
    103f:	90                   	nop
    1040:	f3 0f 1e fa          	endbr64 
    1044:	68 01 00 00 00       	pushq  $0x1
    1049:	f2 e9 d1 ff ff ff    	bnd jmpq 1020 <.plt>
    104f:	90                   	nop
    1050:	f3 0f 1e fa          	endbr64 
    1054:	68 02 00 00 00       	pushq  $0x2
    1059:	f2 e9 c1 ff ff ff    	bnd jmpq 1020 <.plt>
    105f:	90                   	nop
    1060:	f3 0f 1e fa          	endbr64 
    1064:	68 03 00 00 00       	pushq  $0x3
    1069:	f2 e9 b1 ff ff ff    	bnd jmpq 1020 <.plt>
    106f:	90                   	nop
    1070:	f3 0f 1e fa          	endbr64 
    1074:	68 04 00 00 00       	pushq  $0x4
    1079:	f2 e9 a1 ff ff ff    	bnd jmpq 1020 <.plt>
    107f:	90                   	nop
    1080:	f3 0f 1e fa          	endbr64 
    1084:	68 05 00 00 00       	pushq  $0x5
    1089:	f2 e9 91 ff ff ff    	bnd jmpq 1020 <.plt>
    108f:	90                   	nop
    1090:	f3 0f 1e fa          	endbr64 
    1094:	68 06 00 00 00       	pushq  $0x6
    1099:	f2 e9 81 ff ff ff    	bnd jmpq 1020 <.plt>
    109f:	90                   	nop

Disassembly of section .plt.got:

00000000000010a0 <__cxa_finalize@plt>:
    10a0:	f3 0f 1e fa          	endbr64 
    10a4:	f2 ff 25 4d 3f 00 00 	bnd jmpq *0x3f4d(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    10ab:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .plt.sec:

00000000000010b0 <free@plt>:
    10b0:	f3 0f 1e fa          	endbr64 
    10b4:	f2 ff 25 e5 3e 00 00 	bnd jmpq *0x3ee5(%rip)        # 4fa0 <free@GLIBC_2.2.5>
    10bb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010c0 <__stack_chk_fail@plt>:
    10c0:	f3 0f 1e fa          	endbr64 
    10c4:	f2 ff 25 dd 3e 00 00 	bnd jmpq *0x3edd(%rip)        # 4fa8 <__stack_chk_fail@GLIBC_2.4>
    10cb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010d0 <calloc@plt>:
    10d0:	f3 0f 1e fa          	endbr64 
    10d4:	f2 ff 25 d5 3e 00 00 	bnd jmpq *0x3ed5(%rip)        # 4fb0 <calloc@GLIBC_2.2.5>
    10db:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010e0 <memmove@plt>:
    10e0:	f3 0f 1e fa          	endbr64 
    10e4:	f2 ff 25 cd 3e 00 00 	bnd jmpq *0x3ecd(%rip)        # 4fb8 <memmove@GLIBC_2.2.5>
    10eb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

00000000000010f0 <__fprintf_chk@plt>:
    10f0:	f3 0f 1e fa          	endbr64 
    10f4:	f2 ff 25 c5 3e 00 00 	bnd jmpq *0x3ec5(%rip)        # 4fc0 <__fprintf_chk@GLIBC_2.3.4>
    10fb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001100 <rand@plt>:
    1100:	f3 0f 1e fa          	endbr64 
    1104:	f2 ff 25 bd 3e 00 00 	bnd jmpq *0x3ebd(%rip)        # 4fc8 <rand@GLIBC_2.2.5>
    110b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001110 <__sprintf_chk@plt>:
    1110:	f3 0f 1e fa          	endbr64 
    1114:	f2 ff 25 b5 3e 00 00 	bnd jmpq *0x3eb5(%rip)        # 4fd0 <__sprintf_chk@GLIBC_2.3.4>
    111b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

Disassembly of section .text:

0000000000001120 <main>:
    1120:	f3 0f 1e fa          	endbr64 
    1124:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    1129:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
    112d:	41 ff 72 f8          	pushq  -0x8(%r10)
    1131:	55                   	push   %rbp
    1132:	48 89 e5             	mov    %rsp,%rbp
    1135:	41 57                	push   %r15
    1137:	41 56                	push   %r14
    1139:	41 55                	push   %r13
    113b:	41 54                	push   %r12
    113d:	41 52                	push   %r10
    113f:	53                   	push   %rbx
    1140:	48 81 ec 80 01 00 00 	sub    $0x180,%rsp
    1147:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    114e:	00 00 
    1150:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    1154:	31 c0                	xor    %eax,%eax
    1156:	83 ff 01             	cmp    $0x1,%edi
    1159:	7e 44                	jle    119f <main+0x7f>
    115b:	48 8b 35 be 3e 00 00 	mov    0x3ebe(%rip),%rsi        # 5020 <stderr@@GLIBC_2.2.5>
    1162:	48 8d bd 70 fe ff ff 	lea    -0x190(%rbp),%rdi
    1169:	48 8d 15 98 1e 00 00 	lea    0x1e98(%rip),%rdx        # 3008 <_IO_stdin_used+0x8>
    1170:	e8 1b 06 00 00       	callq  1790 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc>
    1175:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    1179:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    1180:	00 00 
    1182:	0f 85 e4 01 00 00    	jne    136c <main+0x24c>
    1188:	48 8d 65 d0          	lea    -0x30(%rbp),%rsp
    118c:	5b                   	pop    %rbx
    118d:	41 5a                	pop    %r10
    118f:	41 5c                	pop    %r12
    1191:	41 5d                	pop    %r13
    1193:	41 5e                	pop    %r14
    1195:	41 5f                	pop    %r15
    1197:	31 c0                	xor    %eax,%eax
    1199:	5d                   	pop    %rbp
    119a:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    119e:	c3                   	retq   
    119f:	31 c0                	xor    %eax,%eax
    11a1:	48 8d bd 50 ff ff ff 	lea    -0xb0(%rbp),%rdi
    11a8:	b9 1c 00 00 00       	mov    $0x1c,%ecx
    11ad:	f3 ab                	rep stos %eax,%es:(%rdi)
    11af:	52                   	push   %rdx
    11b0:	48 8d 0d 8c 1e 00 00 	lea    0x1e8c(%rip),%rcx        # 3043 <_IO_stdin_used+0x43>
    11b7:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
    11be:	6a 02                	pushq  $0x2
    11c0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    11c4:	48 89 c7             	mov    %rax,%rdi
    11c7:	6a 02                	pushq  $0x2
    11c9:	48 89 85 58 fe ff ff 	mov    %rax,-0x1a8(%rbp)
    11d0:	41 b9 04 00 00 00    	mov    $0x4,%r9d
    11d6:	51                   	push   %rcx
    11d7:	4c 8d 05 4d 1e 00 00 	lea    0x1e4d(%rip),%r8        # 302b <_IO_stdin_used+0x2b>
    11de:	48 8d 0d 4f 1e 00 00 	lea    0x1e4f(%rip),%rcx        # 3034 <_IO_stdin_used+0x34>
    11e5:	ba 80 00 00 00       	mov    $0x80,%edx
    11ea:	be 01 00 00 00       	mov    $0x1,%esi
    11ef:	31 c0                	xor    %eax,%eax
    11f1:	c5 f8 29 85 40 ff ff 	vmovaps %xmm0,-0xc0(%rbp)
    11f8:	ff 
    11f9:	e8 12 ff ff ff       	callq  1110 <__sprintf_chk@plt>
    11fe:	48 83 c4 20          	add    $0x20,%rsp
    1202:	be 08 00 00 00       	mov    $0x8,%esi
    1207:	bf 00 01 00 00       	mov    $0x100,%edi
    120c:	e8 bf fe ff ff       	callq  10d0 <calloc@plt>
    1211:	48 89 85 60 fe ff ff 	mov    %rax,-0x1a0(%rbp)
    1218:	41 bd 02 00 00 00    	mov    $0x2,%r13d
    121e:	4c 8d a5 d0 fe ff ff 	lea    -0x130(%rbp),%r12
    1225:	bb 00 00 10 00       	mov    $0x100000,%ebx
    122a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1230:	e8 cb fe ff ff       	callq  1100 <rand@plt>
    1235:	89 85 d0 fe ff ff    	mov    %eax,-0x130(%rbp)
    123b:	e8 c0 fe ff ff       	callq  1100 <rand@plt>
    1240:	89 85 d4 fe ff ff    	mov    %eax,-0x12c(%rbp)
    1246:	e8 b5 fe ff ff       	callq  1100 <rand@plt>
    124b:	89 85 d8 fe ff ff    	mov    %eax,-0x128(%rbp)
    1251:	e8 aa fe ff ff       	callq  1100 <rand@plt>
    1256:	89 85 dc fe ff ff    	mov    %eax,-0x124(%rbp)
    125c:	0f 31                	rdtsc  
    125e:	4c 89 e7             	mov    %r12,%rdi
    1261:	e8 0a 02 00 00       	callq  1470 <_Z7do_sortIL4SORT1EjLj4EN5vsort7bitonicELNS1_17simd_instructionsE2ELNS1_13builtin_usageE2EEvPT0_>
    1266:	0f 31                	rdtsc  
    1268:	ff cb                	dec    %ebx
    126a:	75 c4                	jne    1230 <main+0x110>
    126c:	41 83 fd 01          	cmp    $0x1,%r13d
    1270:	0f 85 eb 00 00 00    	jne    1361 <main+0x241>
    1276:	48 8b 85 60 fe ff ff 	mov    -0x1a0(%rbp),%rax
    127d:	49 89 c7             	mov    %rax,%r15
    1280:	48 05 00 08 00 00    	add    $0x800,%rax
    1286:	48 89 85 68 fe ff ff 	mov    %rax,-0x198(%rbp)
    128d:	0f 1f 00             	nopl   (%rax)
    1290:	41 be 00 00 10 00    	mov    $0x100000,%r14d
    1296:	31 db                	xor    %ebx,%ebx
    1298:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    129f:	00 
    12a0:	e8 5b fe ff ff       	callq  1100 <rand@plt>
    12a5:	89 85 d0 fe ff ff    	mov    %eax,-0x130(%rbp)
    12ab:	e8 50 fe ff ff       	callq  1100 <rand@plt>
    12b0:	89 85 d4 fe ff ff    	mov    %eax,-0x12c(%rbp)
    12b6:	e8 45 fe ff ff       	callq  1100 <rand@plt>
    12bb:	89 85 d8 fe ff ff    	mov    %eax,-0x128(%rbp)
    12c1:	e8 3a fe ff ff       	callq  1100 <rand@plt>
    12c6:	89 85 dc fe ff ff    	mov    %eax,-0x124(%rbp)
    12cc:	0f 31                	rdtsc  
    12ce:	49 89 c5             	mov    %rax,%r13
    12d1:	48 c1 e2 20          	shl    $0x20,%rdx
    12d5:	4c 89 e7             	mov    %r12,%rdi
    12d8:	49 09 d5             	or     %rdx,%r13
    12db:	e8 90 01 00 00       	callq  1470 <_Z7do_sortIL4SORT1EjLj4EN5vsort7bitonicELNS1_17simd_instructionsE2ELNS1_13builtin_usageE2EEvPT0_>
    12e0:	0f 31                	rdtsc  
    12e2:	48 c1 e2 20          	shl    $0x20,%rdx
    12e6:	48 09 d0             	or     %rdx,%rax
    12e9:	4c 29 e8             	sub    %r13,%rax
    12ec:	48 01 c3             	add    %rax,%rbx
    12ef:	41 ff ce             	dec    %r14d
    12f2:	75 ac                	jne    12a0 <main+0x180>
    12f4:	62 f1 ff 08 7b c3    	vcvtusi2sd %rbx,%xmm0,%xmm0
    12fa:	49 83 c7 08          	add    $0x8,%r15
    12fe:	c5 fb 59 05 4a 1d 00 	vmulsd 0x1d4a(%rip),%xmm0,%xmm0        # 3050 <_IO_stdin_used+0x50>
    1305:	00 
    1306:	c4 c1 7b 11 47 f8    	vmovsd %xmm0,-0x8(%r15)
    130c:	4c 3b bd 68 fe ff ff 	cmp    -0x198(%rbp),%r15
    1313:	0f 85 77 ff ff ff    	jne    1290 <main+0x170>
    1319:	48 8b 9d 60 fe ff ff 	mov    -0x1a0(%rbp),%rbx
    1320:	4c 8d a5 70 fe ff ff 	lea    -0x190(%rbp),%r12
    1327:	31 c9                	xor    %ecx,%ecx
    1329:	48 89 de             	mov    %rbx,%rsi
    132c:	4c 89 e7             	mov    %r12,%rdi
    132f:	ba 00 01 00 00       	mov    $0x100,%edx
    1334:	e8 b7 09 00 00       	callq  1cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    1339:	48 8b 8d 58 fe ff ff 	mov    -0x1a8(%rbp),%rcx
    1340:	48 8b 35 d9 3c 00 00 	mov    0x3cd9(%rip),%rsi        # 5020 <stderr@@GLIBC_2.2.5>
    1347:	4c 89 e7             	mov    %r12,%rdi
    134a:	ba 01 00 00 00       	mov    $0x1,%edx
    134f:	e8 7c 04 00 00       	callq  17d0 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_>
    1354:	48 89 df             	mov    %rbx,%rdi
    1357:	e8 54 fd ff ff       	callq  10b0 <free@plt>
    135c:	e9 14 fe ff ff       	jmpq   1175 <main+0x55>
    1361:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    1367:	e9 b9 fe ff ff       	jmpq   1225 <main+0x105>
    136c:	e8 4f fd ff ff       	callq  10c0 <__stack_chk_fail@plt>
    1371:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1378:	00 00 00 
    137b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)

0000000000001380 <_start>:
    1380:	f3 0f 1e fa          	endbr64 
    1384:	31 ed                	xor    %ebp,%ebp
    1386:	49 89 d1             	mov    %rdx,%r9
    1389:	5e                   	pop    %rsi
    138a:	48 89 e2             	mov    %rsp,%rdx
    138d:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    1391:	50                   	push   %rax
    1392:	54                   	push   %rsp
    1393:	4c 8d 05 f6 0e 00 00 	lea    0xef6(%rip),%r8        # 2290 <__libc_csu_fini>
    139a:	48 8d 0d 7f 0e 00 00 	lea    0xe7f(%rip),%rcx        # 2220 <__libc_csu_init>
    13a1:	48 8d 3d 78 fd ff ff 	lea    -0x288(%rip),%rdi        # 1120 <main>
    13a8:	ff 15 32 3c 00 00    	callq  *0x3c32(%rip)        # 4fe0 <__libc_start_main@GLIBC_2.2.5>
    13ae:	f4                   	hlt    
    13af:	90                   	nop

00000000000013b0 <deregister_tm_clones>:
    13b0:	48 8d 3d 59 3c 00 00 	lea    0x3c59(%rip),%rdi        # 5010 <__TMC_END__>
    13b7:	48 8d 05 52 3c 00 00 	lea    0x3c52(%rip),%rax        # 5010 <__TMC_END__>
    13be:	48 39 f8             	cmp    %rdi,%rax
    13c1:	74 15                	je     13d8 <deregister_tm_clones+0x28>
    13c3:	48 8b 05 0e 3c 00 00 	mov    0x3c0e(%rip),%rax        # 4fd8 <_ITM_deregisterTMCloneTable>
    13ca:	48 85 c0             	test   %rax,%rax
    13cd:	74 09                	je     13d8 <deregister_tm_clones+0x28>
    13cf:	ff e0                	jmpq   *%rax
    13d1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    13d8:	c3                   	retq   
    13d9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

00000000000013e0 <register_tm_clones>:
    13e0:	48 8d 3d 29 3c 00 00 	lea    0x3c29(%rip),%rdi        # 5010 <__TMC_END__>
    13e7:	48 8d 35 22 3c 00 00 	lea    0x3c22(%rip),%rsi        # 5010 <__TMC_END__>
    13ee:	48 29 fe             	sub    %rdi,%rsi
    13f1:	48 89 f0             	mov    %rsi,%rax
    13f4:	48 c1 ee 3f          	shr    $0x3f,%rsi
    13f8:	48 c1 f8 03          	sar    $0x3,%rax
    13fc:	48 01 c6             	add    %rax,%rsi
    13ff:	48 d1 fe             	sar    %rsi
    1402:	74 14                	je     1418 <register_tm_clones+0x38>
    1404:	48 8b 05 e5 3b 00 00 	mov    0x3be5(%rip),%rax        # 4ff0 <_ITM_registerTMCloneTable>
    140b:	48 85 c0             	test   %rax,%rax
    140e:	74 08                	je     1418 <register_tm_clones+0x38>
    1410:	ff e0                	jmpq   *%rax
    1412:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1418:	c3                   	retq   
    1419:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001420 <__do_global_dtors_aux>:
    1420:	f3 0f 1e fa          	endbr64 
    1424:	80 3d fd 3b 00 00 00 	cmpb   $0x0,0x3bfd(%rip)        # 5028 <completed.8060>
    142b:	75 2b                	jne    1458 <__do_global_dtors_aux+0x38>
    142d:	55                   	push   %rbp
    142e:	48 83 3d c2 3b 00 00 	cmpq   $0x0,0x3bc2(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    1435:	00 
    1436:	48 89 e5             	mov    %rsp,%rbp
    1439:	74 0c                	je     1447 <__do_global_dtors_aux+0x27>
    143b:	48 8b 3d c6 3b 00 00 	mov    0x3bc6(%rip),%rdi        # 5008 <__dso_handle>
    1442:	e8 59 fc ff ff       	callq  10a0 <__cxa_finalize@plt>
    1447:	e8 64 ff ff ff       	callq  13b0 <deregister_tm_clones>
    144c:	c6 05 d5 3b 00 00 01 	movb   $0x1,0x3bd5(%rip)        # 5028 <completed.8060>
    1453:	5d                   	pop    %rbp
    1454:	c3                   	retq   
    1455:	0f 1f 00             	nopl   (%rax)
    1458:	c3                   	retq   
    1459:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001460 <frame_dummy>:
    1460:	f3 0f 1e fa          	endbr64 
    1464:	e9 77 ff ff ff       	jmpq   13e0 <register_tm_clones>
    1469:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001470 <_Z7do_sortIL4SORT1EjLj4EN5vsort7bitonicELNS1_17simd_instructionsE2ELNS1_13builtin_usageE2EEvPT0_>:
    1470:	f3 0f 1e fa          	endbr64 
    1474:	62 f1 fd 08 6f 17    	vmovdqa64 (%rdi),%xmm2
    147a:	c5 f9 70 ca b1       	vpshufd $0xb1,%xmm2,%xmm1
    147f:	c4 e2 69 3b c1       	vpminud %xmm1,%xmm2,%xmm0
    1484:	c4 e2 69 3f c9       	vpmaxud %xmm1,%xmm2,%xmm1
    1489:	c4 e3 71 02 c8 05    	vpblendd $0x5,%xmm0,%xmm1,%xmm1
    148f:	c5 f9 70 c1 1b       	vpshufd $0x1b,%xmm1,%xmm0
    1494:	c4 e2 71 3b d0       	vpminud %xmm0,%xmm1,%xmm2
    1499:	c4 e2 71 3f c0       	vpmaxud %xmm0,%xmm1,%xmm0
    149e:	c4 e3 79 02 c2 03    	vpblendd $0x3,%xmm2,%xmm0,%xmm0
    14a4:	c5 f9 70 c8 b1       	vpshufd $0xb1,%xmm0,%xmm1
    14a9:	c4 e2 79 3b d1       	vpminud %xmm1,%xmm0,%xmm2
    14ae:	c4 e2 79 3f c1       	vpmaxud %xmm1,%xmm0,%xmm0
    14b3:	c4 e3 79 02 c2 05    	vpblendd $0x5,%xmm2,%xmm0,%xmm0
    14b9:	c5 f8 29 07          	vmovaps %xmm0,(%rdi)
    14bd:	c3                   	retq   
    14be:	66 90                	xchg   %ax,%ax

00000000000014c0 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_>:
    14c0:	f3 0f 1e fa          	endbr64 
    14c4:	41 56                	push   %r14
    14c6:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    14ca:	31 d2                	xor    %edx,%edx
    14cc:	41 55                	push   %r13
    14ce:	41 54                	push   %r12
    14d0:	55                   	push   %rbp
    14d1:	48 89 f5             	mov    %rsi,%rbp
    14d4:	53                   	push   %rbx
    14d5:	48 89 fb             	mov    %rdi,%rbx
    14d8:	48 81 ec 10 02 00 00 	sub    $0x210,%rsp
    14df:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    14e6:	00 00 
    14e8:	48 89 84 24 08 02 00 	mov    %rax,0x208(%rsp)
    14ef:	00 
    14f0:	31 c0                	xor    %eax,%eax
    14f2:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
    14f7:	c5 f8 29 04 24       	vmovaps %xmm0,(%rsp)
    14fc:	89 c1                	mov    %eax,%ecx
    14fe:	83 c0 20             	add    $0x20,%eax
    1501:	48 89 54 0c 10       	mov    %rdx,0x10(%rsp,%rcx,1)
    1506:	48 89 54 0c 18       	mov    %rdx,0x18(%rsp,%rcx,1)
    150b:	48 89 54 0c 20       	mov    %rdx,0x20(%rsp,%rcx,1)
    1510:	48 89 54 0c 28       	mov    %rdx,0x28(%rsp,%rcx,1)
    1515:	3d e0 01 00 00       	cmp    $0x1e0,%eax
    151a:	72 e0                	jb     14fc <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_+0x3c>
    151c:	c5 fb 10 05 64 1d 00 	vmovsd 0x1d64(%rip),%xmm0        # 3288 <_ZN6timersL12units_to_strE+0x28>
    1523:	00 
    1524:	c5 fb 10 4b 48       	vmovsd 0x48(%rbx),%xmm1
    1529:	8b 53 50             	mov    0x50(%rbx),%edx
    152c:	4c 8d 0d 0d 1d 00 00 	lea    0x1d0d(%rip),%r9        # 3240 <_ZN5statsL13possible_fmtsE>
    1533:	48 01 c6             	add    %rax,%rsi
    1536:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    153a:	48 8d 05 1f 1d 00 00 	lea    0x1d1f(%rip),%rax        # 3260 <_ZN6timersL12units_to_strE>
    1541:	48 8d 04 d0          	lea    (%rax,%rdx,8),%rax
    1545:	c5 fb 10 4b 40       	vmovsd 0x40(%rbx),%xmm1
    154a:	49 8d 51 08          	lea    0x8(%r9),%rdx
    154e:	49 89 d6             	mov    %rdx,%r14
    1551:	4d 0f 43 f1          	cmovae %r9,%r14
    1555:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1559:	c5 fb 10 4b 38       	vmovsd 0x38(%rbx),%xmm1
    155e:	49 89 d5             	mov    %rdx,%r13
    1561:	4d 0f 43 e9          	cmovae %r9,%r13
    1565:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1569:	c5 fb 10 4b 30       	vmovsd 0x30(%rbx),%xmm1
    156e:	49 89 d3             	mov    %rdx,%r11
    1571:	4d 0f 43 d9          	cmovae %r9,%r11
    1575:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1579:	c5 fb 10 4b 28       	vmovsd 0x28(%rbx),%xmm1
    157e:	49 89 d2             	mov    %rdx,%r10
    1581:	4d 0f 43 d1          	cmovae %r9,%r10
    1585:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1589:	c5 fb 10 4b 20       	vmovsd 0x20(%rbx),%xmm1
    158e:	49 89 d0             	mov    %rdx,%r8
    1591:	4d 0f 43 c1          	cmovae %r9,%r8
    1595:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1599:	c5 fb 10 4b 18       	vmovsd 0x18(%rbx),%xmm1
    159e:	48 89 d7             	mov    %rdx,%rdi
    15a1:	49 0f 43 f9          	cmovae %r9,%rdi
    15a5:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    15a9:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    15ae:	48 c7 06 00 00 00 00 	movq   $0x0,(%rsi)
    15b5:	48 c7 46 08 00 00 00 	movq   $0x0,0x8(%rsi)
    15bc:	00 
    15bd:	48 89 d6             	mov    %rdx,%rsi
    15c0:	49 0f 43 f1          	cmovae %r9,%rsi
    15c4:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    15c8:	c5 fb 10 4b 08       	vmovsd 0x8(%rbx),%xmm1
    15cd:	48 89 d1             	mov    %rdx,%rcx
    15d0:	49 0f 43 c9          	cmovae %r9,%rcx
    15d4:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    15d8:	49 89 e4             	mov    %rsp,%r12
    15db:	4c 0f 42 ca          	cmovb  %rdx,%r9
    15df:	48 83 ec 08          	sub    $0x8,%rsp
    15e3:	50                   	push   %rax
    15e4:	ba 00 02 00 00       	mov    $0x200,%edx
    15e9:	41 56                	push   %r14
    15eb:	50                   	push   %rax
    15ec:	41 55                	push   %r13
    15ee:	50                   	push   %rax
    15ef:	41 53                	push   %r11
    15f1:	50                   	push   %rax
    15f2:	41 52                	push   %r10
    15f4:	50                   	push   %rax
    15f5:	41 50                	push   %r8
    15f7:	4c 8b 03             	mov    (%rbx),%r8
    15fa:	50                   	push   %rax
    15fb:	57                   	push   %rdi
    15fc:	4c 89 e7             	mov    %r12,%rdi
    15ff:	50                   	push   %rax
    1600:	56                   	push   %rsi
    1601:	be 01 00 00 00       	mov    $0x1,%esi
    1606:	50                   	push   %rax
    1607:	51                   	push   %rcx
    1608:	48 8d 0d 49 1a 00 00 	lea    0x1a49(%rip),%rcx        # 3058 <_IO_stdin_used+0x58>
    160f:	50                   	push   %rax
    1610:	31 c0                	xor    %eax,%eax
    1612:	e8 f9 fa ff ff       	callq  1110 <__sprintf_chk@plt>
    1617:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    161e:	ff 73 48             	pushq  0x48(%rbx)
    1621:	c5 fb 10 43 08       	vmovsd 0x8(%rbx),%xmm0
    1626:	c5 fb 10 7b 40       	vmovsd 0x40(%rbx),%xmm7
    162b:	c5 fb 10 73 38       	vmovsd 0x38(%rbx),%xmm6
    1630:	c5 fb 10 6b 30       	vmovsd 0x30(%rbx),%xmm5
    1635:	c5 fb 10 63 28       	vmovsd 0x28(%rbx),%xmm4
    163a:	c5 fb 10 5b 20       	vmovsd 0x20(%rbx),%xmm3
    163f:	c5 fb 10 53 18       	vmovsd 0x18(%rbx),%xmm2
    1644:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    1649:	4c 89 e2             	mov    %r12,%rdx
    164c:	be 01 00 00 00       	mov    $0x1,%esi
    1651:	48 89 ef             	mov    %rbp,%rdi
    1654:	b8 08 00 00 00       	mov    $0x8,%eax
    1659:	e8 92 fa ff ff       	callq  10f0 <__fprintf_chk@plt>
    165e:	58                   	pop    %rax
    165f:	5a                   	pop    %rdx
    1660:	48 8b 84 24 08 02 00 	mov    0x208(%rsp),%rax
    1667:	00 
    1668:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    166f:	00 00 
    1671:	75 10                	jne    1683 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_+0x1c3>
    1673:	48 81 c4 10 02 00 00 	add    $0x210,%rsp
    167a:	5b                   	pop    %rbx
    167b:	5d                   	pop    %rbp
    167c:	41 5c                	pop    %r12
    167e:	41 5d                	pop    %r13
    1680:	41 5e                	pop    %r14
    1682:	c3                   	retq   
    1683:	e8 38 fa ff ff       	callq  10c0 <__stack_chk_fail@plt>
    1688:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    168f:	00 

0000000000001690 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_>:
    1690:	f3 0f 1e fa          	endbr64 
    1694:	41 55                	push   %r13
    1696:	4d 89 c5             	mov    %r8,%r13
    1699:	41 54                	push   %r12
    169b:	49 89 d4             	mov    %rdx,%r12
    169e:	48 89 ca             	mov    %rcx,%rdx
    16a1:	55                   	push   %rbp
    16a2:	48 89 fd             	mov    %rdi,%rbp
    16a5:	40 84 f6             	test   %sil,%sil
    16a8:	74 16                	je     16c0 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x30>
    16aa:	5d                   	pop    %rbp
    16ab:	4c 89 e6             	mov    %r12,%rsi
    16ae:	41 5c                	pop    %r12
    16b0:	4c 89 c1             	mov    %r8,%rcx
    16b3:	41 5d                	pop    %r13
    16b5:	e9 06 fe ff ff       	jmpq   14c0 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_>
    16ba:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    16c0:	48 83 f9 01          	cmp    $0x1,%rcx
    16c4:	76 09                	jbe    16cf <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x3f>
    16c6:	4d 85 c0             	test   %r8,%r8
    16c9:	0f 85 81 00 00 00    	jne    1750 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0xc0>
    16cf:	48 83 fa 01          	cmp    $0x1,%rdx
    16d3:	0f 85 8d 00 00 00    	jne    1766 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0xd6>
    16d9:	4d 85 ed             	test   %r13,%r13
    16dc:	74 19                	je     16f7 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x67>
    16de:	4c 89 e9             	mov    %r13,%rcx
    16e1:	48 8d 15 4c 1b 00 00 	lea    0x1b4c(%rip),%rdx        # 3234 <_IO_stdin_used+0x234>
    16e8:	be 01 00 00 00       	mov    $0x1,%esi
    16ed:	4c 89 e7             	mov    %r12,%rdi
    16f0:	31 c0                	xor    %eax,%eax
    16f2:	e8 f9 f9 ff ff       	callq  10f0 <__fprintf_chk@plt>
    16f7:	48 83 ec 08          	sub    $0x8,%rsp
    16fb:	ff 75 48             	pushq  0x48(%rbp)
    16fe:	c5 fb 10 45 08       	vmovsd 0x8(%rbp),%xmm0
    1703:	c5 fb 10 7d 40       	vmovsd 0x40(%rbp),%xmm7
    1708:	c5 fb 10 75 38       	vmovsd 0x38(%rbp),%xmm6
    170d:	c5 fb 10 6d 30       	vmovsd 0x30(%rbp),%xmm5
    1712:	c5 fb 10 65 28       	vmovsd 0x28(%rbp),%xmm4
    1717:	c5 fb 10 5d 20       	vmovsd 0x20(%rbp),%xmm3
    171c:	c5 fb 10 55 18       	vmovsd 0x18(%rbp),%xmm2
    1721:	c5 fb 10 4d 10       	vmovsd 0x10(%rbp),%xmm1
    1726:	48 8b 4d 00          	mov    0x0(%rbp),%rcx
    172a:	48 8d 15 6f 1a 00 00 	lea    0x1a6f(%rip),%rdx        # 31a0 <_IO_stdin_used+0x1a0>
    1731:	4c 89 e7             	mov    %r12,%rdi
    1734:	be 01 00 00 00       	mov    $0x1,%esi
    1739:	b8 08 00 00 00       	mov    $0x8,%eax
    173e:	e8 ad f9 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1743:	58                   	pop    %rax
    1744:	5a                   	pop    %rdx
    1745:	5d                   	pop    %rbp
    1746:	41 5c                	pop    %r12
    1748:	41 5d                	pop    %r13
    174a:	c3                   	retq   
    174b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1750:	48 8d 15 dd 1a 00 00 	lea    0x1add(%rip),%rdx        # 3234 <_IO_stdin_used+0x234>
    1757:	be 01 00 00 00       	mov    $0x1,%esi
    175c:	4c 89 e7             	mov    %r12,%rdi
    175f:	31 c0                	xor    %eax,%eax
    1761:	e8 8a f9 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1766:	48 8d 15 fb 19 00 00 	lea    0x19fb(%rip),%rdx        # 3168 <_IO_stdin_used+0x168>
    176d:	be 01 00 00 00       	mov    $0x1,%esi
    1772:	4c 89 e7             	mov    %r12,%rdi
    1775:	31 c0                	xor    %eax,%eax
    1777:	e8 74 f9 ff ff       	callq  10f0 <__fprintf_chk@plt>
    177c:	e9 58 ff ff ff       	jmpq   16d9 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x49>
    1781:	90                   	nop
    1782:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1789:	00 00 00 00 
    178d:	0f 1f 00             	nopl   (%rax)

0000000000001790 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc>:
    1790:	f3 0f 1e fa          	endbr64 
    1794:	55                   	push   %rbp
    1795:	48 89 f5             	mov    %rsi,%rbp
    1798:	48 85 d2             	test   %rdx,%rdx
    179b:	74 19                	je     17b6 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc+0x26>
    179d:	48 89 d1             	mov    %rdx,%rcx
    17a0:	be 01 00 00 00       	mov    $0x1,%esi
    17a5:	48 8d 15 88 1a 00 00 	lea    0x1a88(%rip),%rdx        # 3234 <_IO_stdin_used+0x234>
    17ac:	48 89 ef             	mov    %rbp,%rdi
    17af:	31 c0                	xor    %eax,%eax
    17b1:	e8 3a f9 ff ff       	callq  10f0 <__fprintf_chk@plt>
    17b6:	48 89 ef             	mov    %rbp,%rdi
    17b9:	48 8d 15 a8 19 00 00 	lea    0x19a8(%rip),%rdx        # 3168 <_IO_stdin_used+0x168>
    17c0:	be 01 00 00 00       	mov    $0x1,%esi
    17c5:	31 c0                	xor    %eax,%eax
    17c7:	5d                   	pop    %rbp
    17c8:	e9 23 f9 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    17cd:	90                   	nop
    17ce:	66 90                	xchg   %ax,%ax

00000000000017d0 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_>:
    17d0:	f3 0f 1e fa          	endbr64 
    17d4:	41 54                	push   %r12
    17d6:	49 89 cc             	mov    %rcx,%r12
    17d9:	55                   	push   %rbp
    17da:	48 89 f5             	mov    %rsi,%rbp
    17dd:	53                   	push   %rbx
    17de:	48 89 fb             	mov    %rdi,%rbx
    17e1:	48 83 fa 01          	cmp    $0x1,%rdx
    17e5:	76 09                	jbe    17f0 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x20>
    17e7:	48 85 c9             	test   %rcx,%rcx
    17ea:	0f 85 80 00 00 00    	jne    1870 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0xa0>
    17f0:	48 83 fa 01          	cmp    $0x1,%rdx
    17f4:	0f 85 8f 00 00 00    	jne    1889 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0xb9>
    17fa:	4d 85 e4             	test   %r12,%r12
    17fd:	74 19                	je     1818 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x48>
    17ff:	4c 89 e1             	mov    %r12,%rcx
    1802:	48 8d 15 2b 1a 00 00 	lea    0x1a2b(%rip),%rdx        # 3234 <_IO_stdin_used+0x234>
    1809:	be 01 00 00 00       	mov    $0x1,%esi
    180e:	48 89 ef             	mov    %rbp,%rdi
    1811:	31 c0                	xor    %eax,%eax
    1813:	e8 d8 f8 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1818:	48 83 ec 08          	sub    $0x8,%rsp
    181c:	ff 73 48             	pushq  0x48(%rbx)
    181f:	c5 fb 10 43 08       	vmovsd 0x8(%rbx),%xmm0
    1824:	c5 fb 10 7b 40       	vmovsd 0x40(%rbx),%xmm7
    1829:	c5 fb 10 73 38       	vmovsd 0x38(%rbx),%xmm6
    182e:	c5 fb 10 6b 30       	vmovsd 0x30(%rbx),%xmm5
    1833:	c5 fb 10 63 28       	vmovsd 0x28(%rbx),%xmm4
    1838:	c5 fb 10 5b 20       	vmovsd 0x20(%rbx),%xmm3
    183d:	c5 fb 10 53 18       	vmovsd 0x18(%rbx),%xmm2
    1842:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    1847:	48 8b 0b             	mov    (%rbx),%rcx
    184a:	48 8d 15 4f 19 00 00 	lea    0x194f(%rip),%rdx        # 31a0 <_IO_stdin_used+0x1a0>
    1851:	48 89 ef             	mov    %rbp,%rdi
    1854:	be 01 00 00 00       	mov    $0x1,%esi
    1859:	b8 08 00 00 00       	mov    $0x8,%eax
    185e:	e8 8d f8 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1863:	58                   	pop    %rax
    1864:	5a                   	pop    %rdx
    1865:	5b                   	pop    %rbx
    1866:	5d                   	pop    %rbp
    1867:	41 5c                	pop    %r12
    1869:	c3                   	retq   
    186a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1870:	48 89 d1             	mov    %rdx,%rcx
    1873:	be 01 00 00 00       	mov    $0x1,%esi
    1878:	48 8d 15 b5 19 00 00 	lea    0x19b5(%rip),%rdx        # 3234 <_IO_stdin_used+0x234>
    187f:	48 89 ef             	mov    %rbp,%rdi
    1882:	31 c0                	xor    %eax,%eax
    1884:	e8 67 f8 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1889:	48 8d 15 d8 18 00 00 	lea    0x18d8(%rip),%rdx        # 3168 <_IO_stdin_used+0x168>
    1890:	be 01 00 00 00       	mov    $0x1,%esi
    1895:	48 89 ef             	mov    %rbp,%rdi
    1898:	31 c0                	xor    %eax,%eax
    189a:	e8 51 f8 ff ff       	callq  10f0 <__fprintf_chk@plt>
    189f:	e9 56 ff ff ff       	jmpq   17fa <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x2a>
    18a4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    18ab:	00 00 00 00 
    18af:	90                   	nop

00000000000018b0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE>:
    18b0:	f3 0f 1e fa          	endbr64 
    18b4:	89 4f 50             	mov    %ecx,0x50(%rdi)
    18b7:	83 fa 01             	cmp    $0x1,%edx
    18ba:	0f 84 d0 01 00 00    	je     1a90 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x1e0>
    18c0:	89 d0                	mov    %edx,%eax
    18c2:	48 89 07             	mov    %rax,(%rdi)
    18c5:	89 d0                	mov    %edx,%eax
    18c7:	d1 e8                	shr    %eax
    18c9:	89 c1                	mov    %eax,%ecx
    18cb:	c5 fb 10 04 ce       	vmovsd (%rsi,%rcx,8),%xmm0
    18d0:	f6 c2 01             	test   $0x1,%dl
    18d3:	0f 85 97 01 00 00    	jne    1a70 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x1c0>
    18d9:	ff c0                	inc    %eax
    18db:	c5 fb 58 04 c6       	vaddsd (%rsi,%rax,8),%xmm0,%xmm0
    18e0:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
    18e7:	00 
    18e8:	c5 fb 59 05 a0 19 00 	vmulsd 0x19a0(%rip),%xmm0,%xmm0        # 3290 <_ZN6timersL12units_to_strE+0x30>
    18ef:	00 
    18f0:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    18f5:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    18f9:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
    18fd:	85 d2                	test   %edx,%edx
    18ff:	74 20                	je     1921 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x71>
    1901:	31 c0                	xor    %eax,%eax
    1903:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
    1907:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    190e:	00 00 
    1910:	c5 eb 58 14 c6       	vaddsd (%rsi,%rax,8),%xmm2,%xmm2
    1915:	48 ff c0             	inc    %rax
    1918:	c5 fb 11 57 10       	vmovsd %xmm2,0x10(%rdi)
    191d:	39 c2                	cmp    %eax,%edx
    191f:	77 ef                	ja     1910 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x60>
    1921:	62 f1 57 08 7b ea    	vcvtusi2sd %edx,%xmm5,%xmm5
    1927:	8d 42 ff             	lea    -0x1(%rdx),%eax
    192a:	48 89 c1             	mov    %rax,%rcx
    192d:	c5 eb 5e d5          	vdivsd %xmm5,%xmm2,%xmm2
    1931:	c5 fb 11 57 10       	vmovsd %xmm2,0x10(%rdi)
    1936:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    193b:	c5 fb 11 4f 18       	vmovsd %xmm1,0x18(%rdi)
    1940:	c5 fb 10 0e          	vmovsd (%rsi),%xmm1
    1944:	c5 fb 11 4f 20       	vmovsd %xmm1,0x20(%rdi)
    1949:	c5 d3 59 0d 47 19 00 	vmulsd 0x1947(%rip),%xmm5,%xmm1        # 3298 <_ZN6timersL12units_to_strE+0x38>
    1950:	00 
    1951:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1957:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    195c:	c5 fb 11 4f 28       	vmovsd %xmm1,0x28(%rdi)
    1961:	c5 d3 59 0d 37 19 00 	vmulsd 0x1937(%rip),%xmm5,%xmm1        # 32a0 <_ZN6timersL12units_to_strE+0x40>
    1968:	00 
    1969:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    196f:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1974:	c5 fb 11 4f 30       	vmovsd %xmm1,0x30(%rdi)
    1979:	c5 d3 59 0d 27 19 00 	vmulsd 0x1927(%rip),%xmm5,%xmm1        # 32a8 <_ZN6timersL12units_to_strE+0x48>
    1980:	00 
    1981:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1987:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    198c:	c5 fb 11 4f 38       	vmovsd %xmm1,0x38(%rdi)
    1991:	83 fa 01             	cmp    $0x1,%edx
    1994:	0f 86 56 01 00 00    	jbe    1af0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x240>
    199a:	83 f9 02             	cmp    $0x2,%ecx
    199d:	0f 86 67 01 00 00    	jbe    1b0a <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x25a>
    19a3:	41 89 d0             	mov    %edx,%r8d
    19a6:	41 c1 e8 02          	shr    $0x2,%r8d
    19aa:	41 ff c8             	dec    %r8d
    19ad:	49 c1 e0 05          	shl    $0x5,%r8
    19b1:	c4 e2 7d 19 f2       	vbroadcastsd %xmm2,%ymm6
    19b6:	48 89 f0             	mov    %rsi,%rax
    19b9:	4e 8d 44 06 20       	lea    0x20(%rsi,%r8,1),%r8
    19be:	c5 f9 28 d8          	vmovapd %xmm0,%xmm3
    19c2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    19c8:	c5 fd 10 38          	vmovupd (%rax),%ymm7
    19cc:	48 83 c0 20          	add    $0x20,%rax
    19d0:	c5 c5 5c c6          	vsubpd %ymm6,%ymm7,%ymm0
    19d4:	c5 fd 59 c0          	vmulpd %ymm0,%ymm0,%ymm0
    19d8:	c5 fb 58 db          	vaddsd %xmm3,%xmm0,%xmm3
    19dc:	c5 f9 15 c8          	vunpckhpd %xmm0,%xmm0,%xmm1
    19e0:	62 f3 fd 28 19 c0 01 	vextractf64x2 $0x1,%ymm0,%xmm0
    19e7:	c5 f3 58 cb          	vaddsd %xmm3,%xmm1,%xmm1
    19eb:	c5 f3 58 c8          	vaddsd %xmm0,%xmm1,%xmm1
    19ef:	c5 f9 15 c0          	vunpckhpd %xmm0,%xmm0,%xmm0
    19f3:	c5 f3 58 d8          	vaddsd %xmm0,%xmm1,%xmm3
    19f7:	4c 39 c0             	cmp    %r8,%rax
    19fa:	75 cc                	jne    19c8 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x118>
    19fc:	89 d0                	mov    %edx,%eax
    19fe:	83 e0 fc             	and    $0xfffffffc,%eax
    1a01:	f6 c2 03             	test   $0x3,%dl
    1a04:	0f 84 d6 00 00 00    	je     1ae0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x230>
    1a0a:	41 89 c0             	mov    %eax,%r8d
    1a0d:	c4 a1 7b 10 04 c6    	vmovsd (%rsi,%r8,8),%xmm0
    1a13:	ff c0                	inc    %eax
    1a15:	c5 fb 5c c2          	vsubsd %xmm2,%xmm0,%xmm0
    1a19:	c4 e2 f9 b9 d8       	vfmadd231sd %xmm0,%xmm0,%xmm3
    1a1e:	39 c2                	cmp    %eax,%edx
    1a20:	0f 86 ba 00 00 00    	jbe    1ae0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x230>
    1a26:	c5 f8 77             	vzeroupper 
    1a29:	41 89 c0             	mov    %eax,%r8d
    1a2c:	c4 a1 7b 10 04 c6    	vmovsd (%rsi,%r8,8),%xmm0
    1a32:	ff c0                	inc    %eax
    1a34:	c5 fb 5c c2          	vsubsd %xmm2,%xmm0,%xmm0
    1a38:	c4 e2 f9 b9 d8       	vfmadd231sd %xmm0,%xmm0,%xmm3
    1a3d:	39 c2                	cmp    %eax,%edx
    1a3f:	76 0e                	jbe    1a4f <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x19f>
    1a41:	c5 fb 10 04 c6       	vmovsd (%rsi,%rax,8),%xmm0
    1a46:	c5 fb 5c d2          	vsubsd %xmm2,%xmm0,%xmm2
    1a4a:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
    1a4f:	62 f1 7f 08 7b c1    	vcvtusi2sd %ecx,%xmm0,%xmm0
    1a55:	c5 e3 5e c0          	vdivsd %xmm0,%xmm3,%xmm0
    1a59:	c5 fb 11 47 40       	vmovsd %xmm0,0x40(%rdi)
    1a5e:	c5 e3 5e c5          	vdivsd %xmm5,%xmm3,%xmm0
    1a62:	c5 fb 11 47 48       	vmovsd %xmm0,0x48(%rdi)
    1a67:	c3                   	retq   
    1a68:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1a6f:	00 
    1a70:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1a75:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
    1a7c:	00 
    1a7d:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    1a81:	e9 7b fe ff ff       	jmpq   1901 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x51>
    1a86:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1a8d:	00 00 00 
    1a90:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1a94:	48 c7 07 01 00 00 00 	movq   $0x1,(%rdi)
    1a9b:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1aa0:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1aa4:	c5 fb 11 47 10       	vmovsd %xmm0,0x10(%rdi)
    1aa9:	c5 fb 11 47 18       	vmovsd %xmm0,0x18(%rdi)
    1aae:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1ab2:	c5 fb 11 47 20       	vmovsd %xmm0,0x20(%rdi)
    1ab7:	c5 fb 11 47 28       	vmovsd %xmm0,0x28(%rdi)
    1abc:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1ac0:	48 c7 47 40 00 00 00 	movq   $0x0,0x40(%rdi)
    1ac7:	00 
    1ac8:	48 c7 47 48 00 00 00 	movq   $0x0,0x48(%rdi)
    1acf:	00 
    1ad0:	c5 fb 11 47 30       	vmovsd %xmm0,0x30(%rdi)
    1ad5:	c5 fb 11 47 38       	vmovsd %xmm0,0x38(%rdi)
    1ada:	c3                   	retq   
    1adb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1ae0:	c5 f8 77             	vzeroupper 
    1ae3:	e9 67 ff ff ff       	jmpq   1a4f <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x19f>
    1ae8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1aef:	00 
    1af0:	48 8b 3d 29 35 00 00 	mov    0x3529(%rip),%rdi        # 5020 <stderr@@GLIBC_2.2.5>
    1af7:	48 8d 15 d2 16 00 00 	lea    0x16d2(%rip),%rdx        # 31d0 <_IO_stdin_used+0x1d0>
    1afe:	be 01 00 00 00       	mov    $0x1,%esi
    1b03:	31 c0                	xor    %eax,%eax
    1b05:	e9 e6 f5 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    1b0a:	c5 fb 10 0e          	vmovsd (%rsi),%xmm1
    1b0e:	b8 01 00 00 00       	mov    $0x1,%eax
    1b13:	c5 f3 5c ca          	vsubsd %xmm2,%xmm1,%xmm1
    1b17:	c5 f9 28 d9          	vmovapd %xmm1,%xmm3
    1b1b:	c4 e2 f9 99 d9       	vfmadd132sd %xmm1,%xmm0,%xmm3
    1b20:	e9 04 ff ff ff       	jmpq   1a29 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x179>
    1b25:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1b2c:	00 00 00 00 

0000000000001b30 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>:
    1b30:	48 89 f0             	mov    %rsi,%rax
    1b33:	48 29 f8             	sub    %rdi,%rax
    1b36:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    1b3c:	0f 8e a6 01 00 00    	jle    1ce8 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x1b8>
    1b42:	41 56                	push   %r14
    1b44:	49 89 d6             	mov    %rdx,%r14
    1b47:	41 55                	push   %r13
    1b49:	4c 8d 6f 08          	lea    0x8(%rdi),%r13
    1b4d:	41 54                	push   %r12
    1b4f:	55                   	push   %rbp
    1b50:	53                   	push   %rbx
    1b51:	48 89 fb             	mov    %rdi,%rbx
    1b54:	48 85 d2             	test   %rdx,%rdx
    1b57:	0f 84 03 01 00 00    	je     1c60 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x130>
    1b5d:	48 c1 f8 04          	sar    $0x4,%rax
    1b61:	48 8d 04 c3          	lea    (%rbx,%rax,8),%rax
    1b65:	c5 fb 10 4b 08       	vmovsd 0x8(%rbx),%xmm1
    1b6a:	c5 fb 10 10          	vmovsd (%rax),%xmm2
    1b6e:	49 ff ce             	dec    %r14
    1b71:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
    1b75:	c5 fb 10 5e f8       	vmovsd -0x8(%rsi),%xmm3
    1b7a:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    1b7e:	76 70                	jbe    1bf0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xc0>
    1b80:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
    1b84:	0f 87 b6 00 00 00    	ja     1c40 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x110>
    1b8a:	c5 f9 2f d9          	vcomisd %xmm1,%xmm3
    1b8e:	77 6c                	ja     1bfc <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xcc>
    1b90:	c5 f1 14 c0          	vunpcklpd %xmm0,%xmm1,%xmm0
    1b94:	c5 f8 11 03          	vmovups %xmm0,(%rbx)
    1b98:	c5 fb 10 46 f8       	vmovsd -0x8(%rsi),%xmm0
    1b9d:	4c 89 ed             	mov    %r13,%rbp
    1ba0:	48 89 f0             	mov    %rsi,%rax
    1ba3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1ba8:	c5 fb 10 55 00       	vmovsd 0x0(%rbp),%xmm2
    1bad:	49 89 ec             	mov    %rbp,%r12
    1bb0:	c5 f9 2f ca          	vcomisd %xmm2,%xmm1
    1bb4:	77 30                	ja     1be6 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xb6>
    1bb6:	48 83 e8 08          	sub    $0x8,%rax
    1bba:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1bbe:	76 0f                	jbe    1bcf <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x9f>
    1bc0:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1bc5:	48 83 e8 08          	sub    $0x8,%rax
    1bc9:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1bcd:	77 f1                	ja     1bc0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x90>
    1bcf:	48 39 e8             	cmp    %rbp,%rax
    1bd2:	76 3c                	jbe    1c10 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xe0>
    1bd4:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    1bd9:	c5 fb 11 10          	vmovsd %xmm2,(%rax)
    1bdd:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1be1:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1be6:	48 83 c5 08          	add    $0x8,%rbp
    1bea:	eb bc                	jmp    1ba8 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x78>
    1bec:	0f 1f 40 00          	nopl   0x0(%rax)
    1bf0:	c5 f9 2f d9          	vcomisd %xmm1,%xmm3
    1bf4:	77 9a                	ja     1b90 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x60>
    1bf6:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
    1bfa:	76 44                	jbe    1c40 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x110>
    1bfc:	c5 fb 11 1b          	vmovsd %xmm3,(%rbx)
    1c00:	c5 fb 11 46 f8       	vmovsd %xmm0,-0x8(%rsi)
    1c05:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1c09:	eb 92                	jmp    1b9d <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x6d>
    1c0b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1c10:	4c 89 f2             	mov    %r14,%rdx
    1c13:	48 89 ef             	mov    %rbp,%rdi
    1c16:	e8 15 ff ff ff       	callq  1b30 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>
    1c1b:	48 89 e8             	mov    %rbp,%rax
    1c1e:	48 29 d8             	sub    %rbx,%rax
    1c21:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    1c27:	0f 8e ac 00 00 00    	jle    1cd9 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x1a9>
    1c2d:	4d 85 f6             	test   %r14,%r14
    1c30:	74 31                	je     1c63 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x133>
    1c32:	48 89 ee             	mov    %rbp,%rsi
    1c35:	e9 23 ff ff ff       	jmpq   1b5d <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x2d>
    1c3a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1c40:	c5 fb 11 13          	vmovsd %xmm2,(%rbx)
    1c44:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    1c48:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1c4c:	c5 fb 10 46 f8       	vmovsd -0x8(%rsi),%xmm0
    1c51:	e9 47 ff ff ff       	jmpq   1b9d <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x6d>
    1c56:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1c5d:	00 00 00 
    1c60:	49 89 f4             	mov    %rsi,%r12
    1c63:	48 c1 f8 03          	sar    $0x3,%rax
    1c67:	4c 8d 68 fe          	lea    -0x2(%rax),%r13
    1c6b:	49 d1 fd             	sar    %r13
    1c6e:	c4 a1 7b 10 04 eb    	vmovsd (%rbx,%r13,8),%xmm0
    1c74:	48 89 c2             	mov    %rax,%rdx
    1c77:	4c 89 ee             	mov    %r13,%rsi
    1c7a:	48 89 df             	mov    %rbx,%rdi
    1c7d:	48 89 c5             	mov    %rax,%rbp
    1c80:	e8 6b 04 00 00       	callq  20f0 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1c85:	49 ff cd             	dec    %r13
    1c88:	c4 a1 7b 10 04 eb    	vmovsd (%rbx,%r13,8),%xmm0
    1c8e:	48 89 ea             	mov    %rbp,%rdx
    1c91:	4c 89 ee             	mov    %r13,%rsi
    1c94:	48 89 df             	mov    %rbx,%rdi
    1c97:	e8 54 04 00 00       	callq  20f0 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1c9c:	4d 85 ed             	test   %r13,%r13
    1c9f:	75 e4                	jne    1c85 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x155>
    1ca1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1ca8:	49 83 ec 08          	sub    $0x8,%r12
    1cac:	4c 89 e5             	mov    %r12,%rbp
    1caf:	48 29 dd             	sub    %rbx,%rbp
    1cb2:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1cb6:	48 89 ea             	mov    %rbp,%rdx
    1cb9:	c4 c1 7b 10 04 24    	vmovsd (%r12),%xmm0
    1cbf:	48 c1 fa 03          	sar    $0x3,%rdx
    1cc3:	31 f6                	xor    %esi,%esi
    1cc5:	c4 c1 7b 11 0c 24    	vmovsd %xmm1,(%r12)
    1ccb:	48 89 df             	mov    %rbx,%rdi
    1cce:	e8 1d 04 00 00       	callq  20f0 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1cd3:	48 83 fd 08          	cmp    $0x8,%rbp
    1cd7:	7f cf                	jg     1ca8 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x178>
    1cd9:	5b                   	pop    %rbx
    1cda:	5d                   	pop    %rbp
    1cdb:	41 5c                	pop    %r12
    1cdd:	41 5d                	pop    %r13
    1cdf:	41 5e                	pop    %r14
    1ce1:	c3                   	retq   
    1ce2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1ce8:	c3                   	retq   
    1ce9:	90                   	nop
    1cea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>:
    1cf0:	f3 0f 1e fa          	endbr64 
    1cf4:	41 57                	push   %r15
    1cf6:	41 56                	push   %r14
    1cf8:	41 55                	push   %r13
    1cfa:	41 54                	push   %r12
    1cfc:	55                   	push   %rbp
    1cfd:	53                   	push   %rbx
    1cfe:	48 83 ec 18          	sub    $0x18,%rsp
    1d02:	48 89 3c 24          	mov    %rdi,(%rsp)
    1d06:	85 d2                	test   %edx,%edx
    1d08:	0f 84 5a 01 00 00    	je     1e68 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x178>
    1d0e:	48 89 f5             	mov    %rsi,%rbp
    1d11:	48 85 f6             	test   %rsi,%rsi
    1d14:	0f 84 4e 01 00 00    	je     1e68 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x178>
    1d1a:	41 89 d4             	mov    %edx,%r12d
    1d1d:	49 c1 e4 03          	shl    $0x3,%r12
    1d21:	4c 89 e0             	mov    %r12,%rax
    1d24:	41 89 d7             	mov    %edx,%r15d
    1d27:	48 c1 f8 03          	sar    $0x3,%rax
    1d2b:	ba 3f 00 00 00       	mov    $0x3f,%edx
    1d30:	f3 48 0f bd c0       	lzcnt  %rax,%rax
    1d35:	29 c2                	sub    %eax,%edx
    1d37:	4e 8d 2c 26          	lea    (%rsi,%r12,1),%r13
    1d3b:	48 63 d2             	movslq %edx,%rdx
    1d3e:	48 01 d2             	add    %rdx,%rdx
    1d41:	4c 89 ee             	mov    %r13,%rsi
    1d44:	48 89 ef             	mov    %rbp,%rdi
    1d47:	41 89 ce             	mov    %ecx,%r14d
    1d4a:	48 8d 5d 08          	lea    0x8(%rbp),%rbx
    1d4e:	e8 dd fd ff ff       	callq  1b30 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>
    1d53:	49 81 fc 80 00 00 00 	cmp    $0x80,%r12
    1d5a:	0f 86 30 01 00 00    	jbe    1e90 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1a0>
    1d60:	4c 8d a5 80 00 00 00 	lea    0x80(%rbp),%r12
    1d67:	eb 3e                	jmp    1da7 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xb7>
    1d69:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1d70:	48 39 dd             	cmp    %rbx,%rbp
    1d73:	74 24                	je     1d99 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xa9>
    1d75:	b8 08 00 00 00       	mov    $0x8,%eax
    1d7a:	48 89 da             	mov    %rbx,%rdx
    1d7d:	48 29 ea             	sub    %rbp,%rdx
    1d80:	48 8d 7c 05 00       	lea    0x0(%rbp,%rax,1),%rdi
    1d85:	48 89 ee             	mov    %rbp,%rsi
    1d88:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    1d8e:	e8 4d f3 ff ff       	callq  10e0 <memmove@plt>
    1d93:	c5 fb 10 44 24 08    	vmovsd 0x8(%rsp),%xmm0
    1d99:	48 83 c3 08          	add    $0x8,%rbx
    1d9d:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    1da2:	49 39 dc             	cmp    %rbx,%r12
    1da5:	74 4d                	je     1df4 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x104>
    1da7:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    1dab:	c5 fb 10 4d 00       	vmovsd 0x0(%rbp),%xmm1
    1db0:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1db4:	77 ba                	ja     1d70 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x80>
    1db6:	c5 fb 10 4b f8       	vmovsd -0x8(%rbx),%xmm1
    1dbb:	48 8d 43 f8          	lea    -0x8(%rbx),%rax
    1dbf:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1dc3:	0f 86 67 01 00 00    	jbe    1f30 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x240>
    1dc9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1dd0:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
    1dd5:	48 89 c2             	mov    %rax,%rdx
    1dd8:	c5 fb 10 48 f8       	vmovsd -0x8(%rax),%xmm1
    1ddd:	48 83 e8 08          	sub    $0x8,%rax
    1de1:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1de5:	77 e9                	ja     1dd0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xe0>
    1de7:	48 83 c3 08          	add    $0x8,%rbx
    1deb:	c5 fb 11 02          	vmovsd %xmm0,(%rdx)
    1def:	49 39 dc             	cmp    %rbx,%r12
    1df2:	75 b3                	jne    1da7 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xb7>
    1df4:	4d 39 e5             	cmp    %r12,%r13
    1df7:	74 4b                	je     1e44 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    1df9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1e00:	c4 c1 7b 10 0c 24    	vmovsd (%r12),%xmm1
    1e06:	c4 c1 7b 10 44 24 f8 	vmovsd -0x8(%r12),%xmm0
    1e0d:	49 8d 44 24 f8       	lea    -0x8(%r12),%rax
    1e12:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1e16:	0f 86 fc 00 00 00    	jbe    1f18 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x228>
    1e1c:	0f 1f 40 00          	nopl   0x0(%rax)
    1e20:	c5 fb 11 40 08       	vmovsd %xmm0,0x8(%rax)
    1e25:	48 89 c2             	mov    %rax,%rdx
    1e28:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1e2d:	48 83 e8 08          	sub    $0x8,%rax
    1e31:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1e35:	77 e9                	ja     1e20 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x130>
    1e37:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
    1e3b:	49 83 c4 08          	add    $0x8,%r12
    1e3f:	4d 39 e5             	cmp    %r12,%r13
    1e42:	75 bc                	jne    1e00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x110>
    1e44:	48 8b 3c 24          	mov    (%rsp),%rdi
    1e48:	48 83 c4 18          	add    $0x18,%rsp
    1e4c:	5b                   	pop    %rbx
    1e4d:	48 89 ee             	mov    %rbp,%rsi
    1e50:	5d                   	pop    %rbp
    1e51:	41 5c                	pop    %r12
    1e53:	41 5d                	pop    %r13
    1e55:	44 89 f1             	mov    %r14d,%ecx
    1e58:	41 5e                	pop    %r14
    1e5a:	44 89 fa             	mov    %r15d,%edx
    1e5d:	41 5f                	pop    %r15
    1e5f:	e9 4c fa ff ff       	jmpq   18b0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE>
    1e64:	0f 1f 40 00          	nopl   0x0(%rax)
    1e68:	48 8b 3d b1 31 00 00 	mov    0x31b1(%rip),%rdi        # 5020 <stderr@@GLIBC_2.2.5>
    1e6f:	48 83 c4 18          	add    $0x18,%rsp
    1e73:	5b                   	pop    %rbx
    1e74:	5d                   	pop    %rbp
    1e75:	41 5c                	pop    %r12
    1e77:	41 5d                	pop    %r13
    1e79:	41 5e                	pop    %r14
    1e7b:	48 8d 15 8e 13 00 00 	lea    0x138e(%rip),%rdx        # 3210 <_IO_stdin_used+0x210>
    1e82:	be 01 00 00 00       	mov    $0x1,%esi
    1e87:	31 c0                	xor    %eax,%eax
    1e89:	41 5f                	pop    %r15
    1e8b:	e9 60 f2 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    1e90:	41 bc 08 00 00 00    	mov    $0x8,%r12d
    1e96:	49 39 dd             	cmp    %rbx,%r13
    1e99:	75 3b                	jne    1ed6 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1e6>
    1e9b:	eb a7                	jmp    1e44 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    1e9d:	0f 1f 00             	nopl   (%rax)
    1ea0:	48 39 dd             	cmp    %rbx,%rbp
    1ea3:	74 1f                	je     1ec4 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1d4>
    1ea5:	48 89 da             	mov    %rbx,%rdx
    1ea8:	48 29 ea             	sub    %rbp,%rdx
    1eab:	4a 8d 7c 25 00       	lea    0x0(%rbp,%r12,1),%rdi
    1eb0:	48 89 ee             	mov    %rbp,%rsi
    1eb3:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    1eb9:	e8 22 f2 ff ff       	callq  10e0 <memmove@plt>
    1ebe:	c5 fb 10 44 24 08    	vmovsd 0x8(%rsp),%xmm0
    1ec4:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    1ec9:	48 83 c3 08          	add    $0x8,%rbx
    1ecd:	49 39 dd             	cmp    %rbx,%r13
    1ed0:	0f 84 6e ff ff ff    	je     1e44 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    1ed6:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    1eda:	c5 fb 10 4d 00       	vmovsd 0x0(%rbp),%xmm1
    1edf:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1ee3:	77 bb                	ja     1ea0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1b0>
    1ee5:	c5 fb 10 4b f8       	vmovsd -0x8(%rbx),%xmm1
    1eea:	48 8d 43 f8          	lea    -0x8(%rbx),%rax
    1eee:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1ef2:	76 4c                	jbe    1f40 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x250>
    1ef4:	0f 1f 40 00          	nopl   0x0(%rax)
    1ef8:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
    1efd:	48 89 c2             	mov    %rax,%rdx
    1f00:	c5 fb 10 48 f8       	vmovsd -0x8(%rax),%xmm1
    1f05:	48 83 e8 08          	sub    $0x8,%rax
    1f09:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1f0d:	77 e9                	ja     1ef8 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x208>
    1f0f:	c5 fb 11 02          	vmovsd %xmm0,(%rdx)
    1f13:	eb b4                	jmp    1ec9 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1d9>
    1f15:	0f 1f 00             	nopl   (%rax)
    1f18:	4c 89 e2             	mov    %r12,%rdx
    1f1b:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
    1f1f:	49 83 c4 08          	add    $0x8,%r12
    1f23:	e9 17 ff ff ff       	jmpq   1e3f <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x14f>
    1f28:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1f2f:	00 
    1f30:	48 89 da             	mov    %rbx,%rdx
    1f33:	e9 af fe ff ff       	jmpq   1de7 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xf7>
    1f38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1f3f:	00 
    1f40:	48 89 da             	mov    %rbx,%rdx
    1f43:	eb ca                	jmp    1f0f <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x21f>
    1f45:	90                   	nop
    1f46:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1f4d:	00 00 00 

0000000000001f50 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE>:
    1f50:	f3 0f 1e fa          	endbr64 
    1f54:	85 d2                	test   %edx,%edx
    1f56:	0f 84 84 00 00 00    	je     1fe0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    1f5c:	8d 42 ff             	lea    -0x1(%rdx),%eax
    1f5f:	83 f8 02             	cmp    $0x2,%eax
    1f62:	0f 86 88 00 00 00    	jbe    1ff0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0xa0>
    1f68:	41 89 d0             	mov    %edx,%r8d
    1f6b:	41 c1 e8 02          	shr    $0x2,%r8d
    1f6f:	41 ff c8             	dec    %r8d
    1f72:	49 c1 e0 05          	shl    $0x5,%r8
    1f76:	48 89 f0             	mov    %rsi,%rax
    1f79:	4e 8d 44 06 20       	lea    0x20(%rsi,%r8,1),%r8
    1f7e:	66 90                	xchg   %ax,%ax
    1f80:	62 f1 fe 28 7a 00    	vcvtuqq2pd (%rax),%ymm0
    1f86:	48 83 c0 20          	add    $0x20,%rax
    1f8a:	c5 fd 11 40 e0       	vmovupd %ymm0,-0x20(%rax)
    1f8f:	4c 39 c0             	cmp    %r8,%rax
    1f92:	75 ec                	jne    1f80 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x30>
    1f94:	89 d0                	mov    %edx,%eax
    1f96:	83 e0 fc             	and    $0xfffffffc,%eax
    1f99:	f6 c2 03             	test   $0x3,%dl
    1f9c:	74 4a                	je     1fe8 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x98>
    1f9e:	c5 f8 77             	vzeroupper 
    1fa1:	41 89 c0             	mov    %eax,%r8d
    1fa4:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    1fa8:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    1fae:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    1fb3:	44 8d 40 01          	lea    0x1(%rax),%r8d
    1fb7:	44 39 c2             	cmp    %r8d,%edx
    1fba:	76 24                	jbe    1fe0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    1fbc:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    1fc0:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    1fc6:	83 c0 02             	add    $0x2,%eax
    1fc9:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    1fce:	39 c2                	cmp    %eax,%edx
    1fd0:	76 0e                	jbe    1fe0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    1fd2:	48 8d 04 c6          	lea    (%rsi,%rax,8),%rax
    1fd6:	62 f1 ff 08 7b 00    	vcvtusi2sdq (%rax),%xmm0,%xmm0
    1fdc:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    1fe0:	e9 0b fd ff ff       	jmpq   1cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    1fe5:	0f 1f 00             	nopl   (%rax)
    1fe8:	c5 f8 77             	vzeroupper 
    1feb:	e9 00 fd ff ff       	jmpq   1cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    1ff0:	31 c0                	xor    %eax,%eax
    1ff2:	eb ad                	jmp    1fa1 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x51>
    1ff4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1ffb:	00 00 00 00 
    1fff:	90                   	nop

0000000000002000 <_ZN5stats9stats_out9get_statsERSt6vectorIdSaIdEEN6timers10time_unitsE>:
    2000:	f3 0f 1e fa          	endbr64 
    2004:	4c 8b 06             	mov    (%rsi),%r8
    2007:	89 d1                	mov    %edx,%ecx
    2009:	48 8b 56 08          	mov    0x8(%rsi),%rdx
    200d:	4c 89 c6             	mov    %r8,%rsi
    2010:	4c 29 c2             	sub    %r8,%rdx
    2013:	48 c1 fa 03          	sar    $0x3,%rdx
    2017:	e9 d4 fc ff ff       	jmpq   1cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    201c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002020 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE>:
    2020:	f3 0f 1e fa          	endbr64 
    2024:	48 89 f0             	mov    %rsi,%rax
    2027:	89 d1                	mov    %edx,%ecx
    2029:	48 8b 36             	mov    (%rsi),%rsi
    202c:	48 8b 50 08          	mov    0x8(%rax),%rdx
    2030:	48 29 f2             	sub    %rsi,%rdx
    2033:	48 c1 fa 03          	sar    $0x3,%rdx
    2037:	85 d2                	test   %edx,%edx
    2039:	0f 84 82 00 00 00    	je     20c1 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    203f:	8d 42 ff             	lea    -0x1(%rdx),%eax
    2042:	83 f8 02             	cmp    $0x2,%eax
    2045:	0f 86 95 00 00 00    	jbe    20e0 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xc0>
    204b:	89 d0                	mov    %edx,%eax
    204d:	c1 e8 02             	shr    $0x2,%eax
    2050:	48 c1 e0 05          	shl    $0x5,%rax
    2054:	49 89 f0             	mov    %rsi,%r8
    2057:	48 01 f0             	add    %rsi,%rax
    205a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2060:	62 d1 fe 28 7a 00    	vcvtuqq2pd (%r8),%ymm0
    2066:	49 83 c0 20          	add    $0x20,%r8
    206a:	c4 c1 7d 11 40 e0    	vmovupd %ymm0,-0x20(%r8)
    2070:	49 39 c0             	cmp    %rax,%r8
    2073:	75 eb                	jne    2060 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0x40>
    2075:	89 d0                	mov    %edx,%eax
    2077:	83 e0 fc             	and    $0xfffffffc,%eax
    207a:	f6 c2 03             	test   $0x3,%dl
    207d:	74 51                	je     20d0 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xb0>
    207f:	c5 f8 77             	vzeroupper 
    2082:	41 89 c0             	mov    %eax,%r8d
    2085:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    2089:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    208f:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    2094:	44 8d 40 01          	lea    0x1(%rax),%r8d
    2098:	44 39 c2             	cmp    %r8d,%edx
    209b:	76 24                	jbe    20c1 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    209d:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    20a1:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    20a7:	83 c0 02             	add    $0x2,%eax
    20aa:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    20af:	39 c2                	cmp    %eax,%edx
    20b1:	76 0e                	jbe    20c1 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    20b3:	48 8d 04 c6          	lea    (%rsi,%rax,8),%rax
    20b7:	62 f1 ff 08 7b 00    	vcvtusi2sdq (%rax),%xmm0,%xmm0
    20bd:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    20c1:	e9 2a fc ff ff       	jmpq   1cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    20c6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    20cd:	00 00 00 
    20d0:	c5 f8 77             	vzeroupper 
    20d3:	e9 18 fc ff ff       	jmpq   1cf0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    20d8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    20df:	00 
    20e0:	31 c0                	xor    %eax,%eax
    20e2:	eb 9e                	jmp    2082 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0x62>
    20e4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    20eb:	00 00 00 
    20ee:	66 90                	xchg   %ax,%ax

00000000000020f0 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>:
    20f0:	f3 0f 1e fa          	endbr64 
    20f4:	48 8d 42 ff          	lea    -0x1(%rdx),%rax
    20f8:	49 89 c2             	mov    %rax,%r10
    20fb:	49 c1 ea 3f          	shr    $0x3f,%r10
    20ff:	49 01 c2             	add    %rax,%r10
    2102:	49 89 d3             	mov    %rdx,%r11
    2105:	49 d1 fa             	sar    %r10
    2108:	41 83 e3 01          	and    $0x1,%r11d
    210c:	4c 39 d6             	cmp    %r10,%rsi
    210f:	0f 8d b3 00 00 00    	jge    21c8 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xd8>
    2115:	49 89 f1             	mov    %rsi,%r9
    2118:	eb 20                	jmp    213a <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x4a>
    211a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2120:	48 ff c8             	dec    %rax
    2123:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    2127:	c4 c1 7b 10 08       	vmovsd (%r8),%xmm1
    212c:	c4 a1 7b 11 0c cf    	vmovsd %xmm1,(%rdi,%r9,8)
    2132:	49 39 c2             	cmp    %rax,%r10
    2135:	7e 2f                	jle    2166 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x76>
    2137:	49 89 c1             	mov    %rax,%r9
    213a:	49 8d 49 01          	lea    0x1(%r9),%rcx
    213e:	48 8d 04 09          	lea    (%rcx,%rcx,1),%rax
    2142:	48 c1 e1 04          	shl    $0x4,%rcx
    2146:	4c 8d 04 0f          	lea    (%rdi,%rcx,1),%r8
    214a:	c4 c1 7b 10 08       	vmovsd (%r8),%xmm1
    214f:	c5 fb 10 54 0f f8    	vmovsd -0x8(%rdi,%rcx,1),%xmm2
    2155:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
    2159:	77 c5                	ja     2120 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x30>
    215b:	c4 a1 7b 11 0c cf    	vmovsd %xmm1,(%rdi,%r9,8)
    2161:	49 39 c2             	cmp    %rax,%r10
    2164:	7f d1                	jg     2137 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x47>
    2166:	4d 85 db             	test   %r11,%r11
    2169:	74 6d                	je     21d8 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xe8>
    216b:	48 8d 50 ff          	lea    -0x1(%rax),%rdx
    216f:	48 89 d1             	mov    %rdx,%rcx
    2172:	48 c1 e9 3f          	shr    $0x3f,%rcx
    2176:	48 01 d1             	add    %rdx,%rcx
    2179:	48 d1 f9             	sar    %rcx
    217c:	48 39 f0             	cmp    %rsi,%rax
    217f:	7f 2b                	jg     21ac <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xbc>
    2181:	eb 3c                	jmp    21bf <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xcf>
    2183:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2188:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
    218c:	48 89 d0             	mov    %rdx,%rax
    218f:	48 c1 e8 3f          	shr    $0x3f,%rax
    2193:	48 01 d0             	add    %rdx,%rax
    2196:	48 d1 f8             	sar    %rax
    2199:	48 89 c2             	mov    %rax,%rdx
    219c:	c4 c1 7b 11 08       	vmovsd %xmm1,(%r8)
    21a1:	48 89 c8             	mov    %rcx,%rax
    21a4:	48 39 ce             	cmp    %rcx,%rsi
    21a7:	7d 67                	jge    2210 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x120>
    21a9:	48 89 d1             	mov    %rdx,%rcx
    21ac:	4c 8d 0c cf          	lea    (%rdi,%rcx,8),%r9
    21b0:	c4 c1 7b 10 09       	vmovsd (%r9),%xmm1
    21b5:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    21b9:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    21bd:	77 c9                	ja     2188 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x98>
    21bf:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    21c4:	c3                   	retq   
    21c5:	0f 1f 00             	nopl   (%rax)
    21c8:	4c 8d 04 f7          	lea    (%rdi,%rsi,8),%r8
    21cc:	4d 85 db             	test   %r11,%r11
    21cf:	75 ee                	jne    21bf <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xcf>
    21d1:	48 89 f0             	mov    %rsi,%rax
    21d4:	0f 1f 40 00          	nopl   0x0(%rax)
    21d8:	48 83 ea 02          	sub    $0x2,%rdx
    21dc:	48 89 d1             	mov    %rdx,%rcx
    21df:	48 c1 e9 3f          	shr    $0x3f,%rcx
    21e3:	48 01 ca             	add    %rcx,%rdx
    21e6:	48 d1 fa             	sar    %rdx
    21e9:	48 39 c2             	cmp    %rax,%rdx
    21ec:	0f 85 79 ff ff ff    	jne    216b <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x7b>
    21f2:	48 8d 44 00 02       	lea    0x2(%rax,%rax,1),%rax
    21f7:	c5 fb 10 4c c7 f8    	vmovsd -0x8(%rdi,%rax,8),%xmm1
    21fd:	48 ff c8             	dec    %rax
    2200:	c4 c1 7b 11 08       	vmovsd %xmm1,(%r8)
    2205:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    2209:	e9 5d ff ff ff       	jmpq   216b <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x7b>
    220e:	66 90                	xchg   %ax,%ax
    2210:	4d 89 c8             	mov    %r9,%r8
    2213:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    2218:	c3                   	retq   
    2219:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002220 <__libc_csu_init>:
    2220:	f3 0f 1e fa          	endbr64 
    2224:	41 57                	push   %r15
    2226:	4c 8d 3d 5b 2b 00 00 	lea    0x2b5b(%rip),%r15        # 4d88 <__frame_dummy_init_array_entry>
    222d:	41 56                	push   %r14
    222f:	49 89 d6             	mov    %rdx,%r14
    2232:	41 55                	push   %r13
    2234:	49 89 f5             	mov    %rsi,%r13
    2237:	41 54                	push   %r12
    2239:	41 89 fc             	mov    %edi,%r12d
    223c:	55                   	push   %rbp
    223d:	48 8d 2d 4c 2b 00 00 	lea    0x2b4c(%rip),%rbp        # 4d90 <__do_global_dtors_aux_fini_array_entry>
    2244:	53                   	push   %rbx
    2245:	4c 29 fd             	sub    %r15,%rbp
    2248:	48 83 ec 08          	sub    $0x8,%rsp
    224c:	e8 af ed ff ff       	callq  1000 <_init>
    2251:	48 c1 fd 03          	sar    $0x3,%rbp
    2255:	74 1f                	je     2276 <__libc_csu_init+0x56>
    2257:	31 db                	xor    %ebx,%ebx
    2259:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2260:	4c 89 f2             	mov    %r14,%rdx
    2263:	4c 89 ee             	mov    %r13,%rsi
    2266:	44 89 e7             	mov    %r12d,%edi
    2269:	41 ff 14 df          	callq  *(%r15,%rbx,8)
    226d:	48 83 c3 01          	add    $0x1,%rbx
    2271:	48 39 dd             	cmp    %rbx,%rbp
    2274:	75 ea                	jne    2260 <__libc_csu_init+0x40>
    2276:	48 83 c4 08          	add    $0x8,%rsp
    227a:	5b                   	pop    %rbx
    227b:	5d                   	pop    %rbp
    227c:	41 5c                	pop    %r12
    227e:	41 5d                	pop    %r13
    2280:	41 5e                	pop    %r14
    2282:	41 5f                	pop    %r15
    2284:	c3                   	retq   
    2285:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    228c:	00 00 00 00 

0000000000002290 <__libc_csu_fini>:
    2290:	f3 0f 1e fa          	endbr64 
    2294:	c3                   	retq   

Disassembly of section .fini:

0000000000002298 <_fini>:
    2298:	f3 0f 1e fa          	endbr64 
    229c:	48 83 ec 08          	sub    $0x8,%rsp
    22a0:	48 83 c4 08          	add    $0x8,%rsp
    22a4:	c3                   	retq   
