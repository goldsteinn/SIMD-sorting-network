
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

0000000000001140 <main>:
    1140:	f3 0f 1e fa          	endbr64 
    1144:	4c 8d 54 24 08       	lea    0x8(%rsp),%r10
    1149:	48 83 e4 c0          	and    $0xffffffffffffffc0,%rsp
    114d:	41 ff 72 f8          	pushq  -0x8(%r10)
    1151:	55                   	push   %rbp
    1152:	48 89 e5             	mov    %rsp,%rbp
    1155:	41 57                	push   %r15
    1157:	41 56                	push   %r14
    1159:	41 55                	push   %r13
    115b:	41 54                	push   %r12
    115d:	41 52                	push   %r10
    115f:	53                   	push   %rbx
    1160:	48 81 ec 80 01 00 00 	sub    $0x180,%rsp
    1167:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    116e:	00 00 
    1170:	48 89 45 c8          	mov    %rax,-0x38(%rbp)
    1174:	31 c0                	xor    %eax,%eax
    1176:	83 ff 01             	cmp    $0x1,%edi
    1179:	7e 44                	jle    11bf <main+0x7f>
    117b:	48 8b 35 9e 3e 00 00 	mov    0x3e9e(%rip),%rsi        # 5020 <stderr@@GLIBC_2.2.5>
    1182:	48 8d bd 70 fe ff ff 	lea    -0x190(%rbp),%rdi
    1189:	48 8d 15 78 1e 00 00 	lea    0x1e78(%rip),%rdx        # 3008 <_IO_stdin_used+0x8>
    1190:	e8 0b 08 00 00       	callq  19a0 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc>
    1195:	48 8b 45 c8          	mov    -0x38(%rbp),%rax
    1199:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    11a0:	00 00 
    11a2:	0f 85 ff 01 00 00    	jne    13a7 <main+0x267>
    11a8:	48 8d 65 d0          	lea    -0x30(%rbp),%rsp
    11ac:	5b                   	pop    %rbx
    11ad:	41 5a                	pop    %r10
    11af:	41 5c                	pop    %r12
    11b1:	41 5d                	pop    %r13
    11b3:	41 5e                	pop    %r14
    11b5:	41 5f                	pop    %r15
    11b7:	31 c0                	xor    %eax,%eax
    11b9:	5d                   	pop    %rbp
    11ba:	49 8d 62 f8          	lea    -0x8(%r10),%rsp
    11be:	c3                   	retq   
    11bf:	31 c0                	xor    %eax,%eax
    11c1:	48 8d bd 50 ff ff ff 	lea    -0xb0(%rbp),%rdi
    11c8:	b9 1c 00 00 00       	mov    $0x1c,%ecx
    11cd:	f3 ab                	rep stos %eax,%es:(%rdi)
    11cf:	52                   	push   %rdx
    11d0:	48 8d 0d 6b 1e 00 00 	lea    0x1e6b(%rip),%rcx        # 3042 <_IO_stdin_used+0x42>
    11d7:	48 8d 85 40 ff ff ff 	lea    -0xc0(%rbp),%rax
    11de:	6a 00                	pushq  $0x0
    11e0:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    11e4:	48 89 c7             	mov    %rax,%rdi
    11e7:	6a 01                	pushq  $0x1
    11e9:	48 89 85 58 fe ff ff 	mov    %rax,-0x1a8(%rbp)
    11f0:	41 b9 11 00 00 00    	mov    $0x11,%r9d
    11f6:	51                   	push   %rcx
    11f7:	4c 8d 05 2d 1e 00 00 	lea    0x1e2d(%rip),%r8        # 302b <_IO_stdin_used+0x2b>
    11fe:	48 8d 0d 2e 1e 00 00 	lea    0x1e2e(%rip),%rcx        # 3033 <_IO_stdin_used+0x33>
    1205:	ba 80 00 00 00       	mov    $0x80,%edx
    120a:	be 01 00 00 00       	mov    $0x1,%esi
    120f:	31 c0                	xor    %eax,%eax
    1211:	c5 f8 29 85 40 ff ff 	vmovaps %xmm0,-0xc0(%rbp)
    1218:	ff 
    1219:	e8 f2 fe ff ff       	callq  1110 <__sprintf_chk@plt>
    121e:	48 83 c4 20          	add    $0x20,%rsp
    1222:	be 08 00 00 00       	mov    $0x8,%esi
    1227:	bf 00 01 00 00       	mov    $0x100,%edi
    122c:	e8 9f fe ff ff       	callq  10d0 <calloc@plt>
    1231:	48 89 85 60 fe ff ff 	mov    %rax,-0x1a0(%rbp)
    1238:	41 bd 02 00 00 00    	mov    $0x2,%r13d
    123e:	4c 8d a5 d0 fe ff ff 	lea    -0x130(%rbp),%r12
    1245:	bb 00 00 10 00       	mov    $0x100000,%ebx
    124a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1250:	e8 ab fe ff ff       	callq  1100 <rand@plt>
    1255:	89 85 d0 fe ff ff    	mov    %eax,-0x130(%rbp)
    125b:	e8 a0 fe ff ff       	callq  1100 <rand@plt>
    1260:	89 85 d4 fe ff ff    	mov    %eax,-0x12c(%rbp)
    1266:	e8 95 fe ff ff       	callq  1100 <rand@plt>
    126b:	89 85 d8 fe ff ff    	mov    %eax,-0x128(%rbp)
    1271:	e8 8a fe ff ff       	callq  1100 <rand@plt>
    1276:	89 85 dc fe ff ff    	mov    %eax,-0x124(%rbp)
    127c:	e8 7f fe ff ff       	callq  1100 <rand@plt>
    1281:	89 85 e0 fe ff ff    	mov    %eax,-0x120(%rbp)
    1287:	0f 31                	rdtsc  
    1289:	4c 89 e7             	mov    %r12,%rdi
    128c:	e8 2f 02 00 00       	callq  14c0 <_Z7do_sortIL4SORT1EhLj17EN5vsort7bitonicELNS1_17simd_instructionsE1ELNS1_13builtin_usageE0EEvPT0_>
    1291:	0f 31                	rdtsc  
    1293:	ff cb                	dec    %ebx
    1295:	75 b9                	jne    1250 <main+0x110>
    1297:	41 83 fd 01          	cmp    $0x1,%r13d
    129b:	0f 85 fb 00 00 00    	jne    139c <main+0x25c>
    12a1:	48 8b 85 60 fe ff ff 	mov    -0x1a0(%rbp),%rax
    12a8:	49 89 c7             	mov    %rax,%r15
    12ab:	48 05 00 08 00 00    	add    $0x800,%rax
    12b1:	48 89 85 68 fe ff ff 	mov    %rax,-0x198(%rbp)
    12b8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    12bf:	00 
    12c0:	41 be 00 00 10 00    	mov    $0x100000,%r14d
    12c6:	31 db                	xor    %ebx,%ebx
    12c8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    12cf:	00 
    12d0:	e8 2b fe ff ff       	callq  1100 <rand@plt>
    12d5:	89 85 d0 fe ff ff    	mov    %eax,-0x130(%rbp)
    12db:	e8 20 fe ff ff       	callq  1100 <rand@plt>
    12e0:	89 85 d4 fe ff ff    	mov    %eax,-0x12c(%rbp)
    12e6:	e8 15 fe ff ff       	callq  1100 <rand@plt>
    12eb:	89 85 d8 fe ff ff    	mov    %eax,-0x128(%rbp)
    12f1:	e8 0a fe ff ff       	callq  1100 <rand@plt>
    12f6:	89 85 dc fe ff ff    	mov    %eax,-0x124(%rbp)
    12fc:	e8 ff fd ff ff       	callq  1100 <rand@plt>
    1301:	89 85 e0 fe ff ff    	mov    %eax,-0x120(%rbp)
    1307:	0f 31                	rdtsc  
    1309:	49 89 c5             	mov    %rax,%r13
    130c:	48 c1 e2 20          	shl    $0x20,%rdx
    1310:	4c 89 e7             	mov    %r12,%rdi
    1313:	49 09 d5             	or     %rdx,%r13
    1316:	e8 a5 01 00 00       	callq  14c0 <_Z7do_sortIL4SORT1EhLj17EN5vsort7bitonicELNS1_17simd_instructionsE1ELNS1_13builtin_usageE0EEvPT0_>
    131b:	0f 31                	rdtsc  
    131d:	48 c1 e2 20          	shl    $0x20,%rdx
    1321:	48 09 d0             	or     %rdx,%rax
    1324:	4c 29 e8             	sub    %r13,%rax
    1327:	48 01 c3             	add    %rax,%rbx
    132a:	41 ff ce             	dec    %r14d
    132d:	75 a1                	jne    12d0 <main+0x190>
    132f:	62 f1 ff 08 7b c3    	vcvtusi2sd %rbx,%xmm0,%xmm0
    1335:	49 83 c7 08          	add    $0x8,%r15
    1339:	c5 fb 59 05 bf 20 00 	vmulsd 0x20bf(%rip),%xmm0,%xmm0        # 3400 <_IO_stdin_used+0x400>
    1340:	00 
    1341:	c4 c1 7b 11 47 f8    	vmovsd %xmm0,-0x8(%r15)
    1347:	4c 3b bd 68 fe ff ff 	cmp    -0x198(%rbp),%r15
    134e:	0f 85 6c ff ff ff    	jne    12c0 <main+0x180>
    1354:	48 8b 9d 60 fe ff ff 	mov    -0x1a0(%rbp),%rbx
    135b:	4c 8d a5 70 fe ff ff 	lea    -0x190(%rbp),%r12
    1362:	31 c9                	xor    %ecx,%ecx
    1364:	48 89 de             	mov    %rbx,%rsi
    1367:	4c 89 e7             	mov    %r12,%rdi
    136a:	ba 00 01 00 00       	mov    $0x100,%edx
    136f:	e8 8c 0b 00 00       	callq  1f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    1374:	48 8b 8d 58 fe ff ff 	mov    -0x1a8(%rbp),%rcx
    137b:	48 8b 35 9e 3c 00 00 	mov    0x3c9e(%rip),%rsi        # 5020 <stderr@@GLIBC_2.2.5>
    1382:	4c 89 e7             	mov    %r12,%rdi
    1385:	ba 01 00 00 00       	mov    $0x1,%edx
    138a:	e8 51 06 00 00       	callq  19e0 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_>
    138f:	48 89 df             	mov    %rbx,%rdi
    1392:	e8 19 fd ff ff       	callq  10b0 <free@plt>
    1397:	e9 f9 fd ff ff       	jmpq   1195 <main+0x55>
    139c:	41 bd 01 00 00 00    	mov    $0x1,%r13d
    13a2:	e9 9e fe ff ff       	jmpq   1245 <main+0x105>
    13a7:	e8 14 fd ff ff       	callq  10c0 <__stack_chk_fail@plt>
    13ac:	0f 1f 40 00          	nopl   0x0(%rax)

00000000000013b0 <_start>:
    13b0:	f3 0f 1e fa          	endbr64 
    13b4:	31 ed                	xor    %ebp,%ebp
    13b6:	49 89 d1             	mov    %rdx,%r9
    13b9:	5e                   	pop    %rsi
    13ba:	48 89 e2             	mov    %rsp,%rdx
    13bd:	48 83 e4 f0          	and    $0xfffffffffffffff0,%rsp
    13c1:	50                   	push   %rax
    13c2:	54                   	push   %rsp
    13c3:	4c 8d 05 d6 10 00 00 	lea    0x10d6(%rip),%r8        # 24a0 <__libc_csu_fini>
    13ca:	48 8d 0d 5f 10 00 00 	lea    0x105f(%rip),%rcx        # 2430 <__libc_csu_init>
    13d1:	48 8d 3d 68 fd ff ff 	lea    -0x298(%rip),%rdi        # 1140 <main>
    13d8:	ff 15 02 3c 00 00    	callq  *0x3c02(%rip)        # 4fe0 <__libc_start_main@GLIBC_2.2.5>
    13de:	f4                   	hlt    
    13df:	90                   	nop

00000000000013e0 <deregister_tm_clones>:
    13e0:	48 8d 3d 29 3c 00 00 	lea    0x3c29(%rip),%rdi        # 5010 <__TMC_END__>
    13e7:	48 8d 05 22 3c 00 00 	lea    0x3c22(%rip),%rax        # 5010 <__TMC_END__>
    13ee:	48 39 f8             	cmp    %rdi,%rax
    13f1:	74 15                	je     1408 <deregister_tm_clones+0x28>
    13f3:	48 8b 05 de 3b 00 00 	mov    0x3bde(%rip),%rax        # 4fd8 <_ITM_deregisterTMCloneTable>
    13fa:	48 85 c0             	test   %rax,%rax
    13fd:	74 09                	je     1408 <deregister_tm_clones+0x28>
    13ff:	ff e0                	jmpq   *%rax
    1401:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1408:	c3                   	retq   
    1409:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001410 <register_tm_clones>:
    1410:	48 8d 3d f9 3b 00 00 	lea    0x3bf9(%rip),%rdi        # 5010 <__TMC_END__>
    1417:	48 8d 35 f2 3b 00 00 	lea    0x3bf2(%rip),%rsi        # 5010 <__TMC_END__>
    141e:	48 29 fe             	sub    %rdi,%rsi
    1421:	48 89 f0             	mov    %rsi,%rax
    1424:	48 c1 ee 3f          	shr    $0x3f,%rsi
    1428:	48 c1 f8 03          	sar    $0x3,%rax
    142c:	48 01 c6             	add    %rax,%rsi
    142f:	48 d1 fe             	sar    %rsi
    1432:	74 14                	je     1448 <register_tm_clones+0x38>
    1434:	48 8b 05 b5 3b 00 00 	mov    0x3bb5(%rip),%rax        # 4ff0 <_ITM_registerTMCloneTable>
    143b:	48 85 c0             	test   %rax,%rax
    143e:	74 08                	je     1448 <register_tm_clones+0x38>
    1440:	ff e0                	jmpq   *%rax
    1442:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1448:	c3                   	retq   
    1449:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001450 <__do_global_dtors_aux>:
    1450:	f3 0f 1e fa          	endbr64 
    1454:	80 3d cd 3b 00 00 00 	cmpb   $0x0,0x3bcd(%rip)        # 5028 <completed.8060>
    145b:	75 2b                	jne    1488 <__do_global_dtors_aux+0x38>
    145d:	55                   	push   %rbp
    145e:	48 83 3d 92 3b 00 00 	cmpq   $0x0,0x3b92(%rip)        # 4ff8 <__cxa_finalize@GLIBC_2.2.5>
    1465:	00 
    1466:	48 89 e5             	mov    %rsp,%rbp
    1469:	74 0c                	je     1477 <__do_global_dtors_aux+0x27>
    146b:	48 8b 3d 96 3b 00 00 	mov    0x3b96(%rip),%rdi        # 5008 <__dso_handle>
    1472:	e8 29 fc ff ff       	callq  10a0 <__cxa_finalize@plt>
    1477:	e8 64 ff ff ff       	callq  13e0 <deregister_tm_clones>
    147c:	c6 05 a5 3b 00 00 01 	movb   $0x1,0x3ba5(%rip)        # 5028 <completed.8060>
    1483:	5d                   	pop    %rbp
    1484:	c3                   	retq   
    1485:	0f 1f 00             	nopl   (%rax)
    1488:	c3                   	retq   
    1489:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000001490 <frame_dummy>:
    1490:	f3 0f 1e fa          	endbr64 
    1494:	e9 77 ff ff ff       	jmpq   1410 <register_tm_clones>
    1499:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    14a0:	00 00 00 
    14a3:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    14aa:	00 00 00 
    14ad:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    14b4:	00 00 00 
    14b7:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    14be:	00 00 

00000000000014c0 <_Z7do_sortIL4SORT1EhLj17EN5vsort7bitonicELNS1_17simd_instructionsE1ELNS1_13builtin_usageE0EEvPT0_>:
    14c0:	f3 0f 1e fa          	endbr64 
    14c4:	62 f1 fd 28 6f 17    	vmovdqa64 (%rdi),%ymm2
    14ca:	c4 e2 6d 00 05 8d 1b 	vpshufb 0x1b8d(%rip),%ymm2,%ymm0        # 3060 <_IO_stdin_used+0x60>
    14d1:	00 00 
    14d3:	c4 e2 6d 00 1d a4 1b 	vpshufb 0x1ba4(%rip),%ymm2,%ymm3        # 3080 <_IO_stdin_used+0x80>
    14da:	00 00 
    14dc:	c4 e3 fd 00 c0 4e    	vpermq $0x4e,%ymm0,%ymm0
    14e2:	c5 e5 eb c0          	vpor   %ymm0,%ymm3,%ymm0
    14e6:	c5 fd da ca          	vpminub %ymm2,%ymm0,%ymm1
    14ea:	c5 fd de c2          	vpmaxub %ymm2,%ymm0,%ymm0
    14ee:	62 f1 fd 28 6f 15 a8 	vmovdqa64 0x1ba8(%rip),%ymm2        # 30a0 <_IO_stdin_used+0xa0>
    14f5:	1b 00 00 
    14f8:	c4 e3 7d 4c d1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm2
    14fe:	c4 e2 6d 00 0d b9 1b 	vpshufb 0x1bb9(%rip),%ymm2,%ymm1        # 30c0 <_IO_stdin_used+0xc0>
    1505:	00 00 
    1507:	c4 e3 fd 00 c1 4e    	vpermq $0x4e,%ymm1,%ymm0
    150d:	c4 e2 6d 00 0d ca 1b 	vpshufb 0x1bca(%rip),%ymm2,%ymm1        # 30e0 <_IO_stdin_used+0xe0>
    1514:	00 00 
    1516:	c5 f5 eb c8          	vpor   %ymm0,%ymm1,%ymm1
    151a:	c5 ed da c1          	vpminub %ymm1,%ymm2,%ymm0
    151e:	c5 ed de d1          	vpmaxub %ymm1,%ymm2,%ymm2
    1522:	62 f1 fd 28 6f 0d d4 	vmovdqa64 0x1bd4(%rip),%ymm1        # 3100 <_IO_stdin_used+0x100>
    1529:	1b 00 00 
    152c:	c4 e3 6d 4c c8 10    	vpblendvb %ymm1,%ymm0,%ymm2,%ymm1
    1532:	c4 e2 75 00 05 e5 1b 	vpshufb 0x1be5(%rip),%ymm1,%ymm0        # 3120 <_IO_stdin_used+0x120>
    1539:	00 00 
    153b:	c4 e3 fd 00 d0 4e    	vpermq $0x4e,%ymm0,%ymm2
    1541:	c4 e2 75 00 05 f6 1b 	vpshufb 0x1bf6(%rip),%ymm1,%ymm0        # 3140 <_IO_stdin_used+0x140>
    1548:	00 00 
    154a:	c5 fd eb c2          	vpor   %ymm2,%ymm0,%ymm0
    154e:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    1552:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    1556:	62 f1 fd 28 6f 05 00 	vmovdqa64 0x1c00(%rip),%ymm0        # 3160 <_IO_stdin_used+0x160>
    155d:	1c 00 00 
    1560:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    1566:	c4 e2 7d 00 0d 11 1c 	vpshufb 0x1c11(%rip),%ymm0,%ymm1        # 3180 <_IO_stdin_used+0x180>
    156d:	00 00 
    156f:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    1575:	c4 e2 7d 00 0d 22 1c 	vpshufb 0x1c22(%rip),%ymm0,%ymm1        # 31a0 <_IO_stdin_used+0x1a0>
    157c:	00 00 
    157e:	c5 f5 eb ca          	vpor   %ymm2,%ymm1,%ymm1
    1582:	c5 fd da d1          	vpminub %ymm1,%ymm0,%ymm2
    1586:	c5 fd de c1          	vpmaxub %ymm1,%ymm0,%ymm0
    158a:	62 f1 fd 28 6f 0d 2c 	vmovdqa64 0x1c2c(%rip),%ymm1        # 31c0 <_IO_stdin_used+0x1c0>
    1591:	1c 00 00 
    1594:	c4 e3 7d 4c c2 10    	vpblendvb %ymm1,%ymm2,%ymm0,%ymm0
    159a:	c4 e2 7d 00 0d 3d 1c 	vpshufb 0x1c3d(%rip),%ymm0,%ymm1        # 31e0 <_IO_stdin_used+0x1e0>
    15a1:	00 00 
    15a3:	c5 fd da d1          	vpminub %ymm1,%ymm0,%ymm2
    15a7:	c5 fd de c1          	vpmaxub %ymm1,%ymm0,%ymm0
    15ab:	62 f1 fd 28 6f 0d 4b 	vmovdqa64 0x1c4b(%rip),%ymm1        # 3200 <_IO_stdin_used+0x200>
    15b2:	1c 00 00 
    15b5:	c4 e3 7d 4c ca 10    	vpblendvb %ymm1,%ymm2,%ymm0,%ymm1
    15bb:	c4 e2 75 00 05 5c 1c 	vpshufb 0x1c5c(%rip),%ymm1,%ymm0        # 3220 <_IO_stdin_used+0x220>
    15c2:	00 00 
    15c4:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    15c8:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    15cc:	62 f1 fd 28 6f 05 6a 	vmovdqa64 0x1c6a(%rip),%ymm0        # 3240 <_IO_stdin_used+0x240>
    15d3:	1c 00 00 
    15d6:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    15dc:	c4 e2 7d 00 0d 7b 1c 	vpshufb 0x1c7b(%rip),%ymm0,%ymm1        # 3260 <_IO_stdin_used+0x260>
    15e3:	00 00 
    15e5:	c4 e3 fd 00 d1 4e    	vpermq $0x4e,%ymm1,%ymm2
    15eb:	c4 e2 7d 00 0d 8c 1c 	vpshufb 0x1c8c(%rip),%ymm0,%ymm1        # 3280 <_IO_stdin_used+0x280>
    15f2:	00 00 
    15f4:	c5 f5 eb ca          	vpor   %ymm2,%ymm1,%ymm1
    15f8:	c5 fd da d1          	vpminub %ymm1,%ymm0,%ymm2
    15fc:	c5 fd de c1          	vpmaxub %ymm1,%ymm0,%ymm0
    1600:	62 f1 fd 28 6f 0d 96 	vmovdqa64 0x1c96(%rip),%ymm1        # 32a0 <_IO_stdin_used+0x2a0>
    1607:	1c 00 00 
    160a:	c4 e3 7d 4c c2 10    	vpblendvb %ymm1,%ymm2,%ymm0,%ymm0
    1610:	c4 e2 7d 00 15 a7 1c 	vpshufb 0x1ca7(%rip),%ymm0,%ymm2        # 32c0 <_IO_stdin_used+0x2c0>
    1617:	00 00 
    1619:	c5 fd da ca          	vpminub %ymm2,%ymm0,%ymm1
    161d:	c5 fd de c2          	vpmaxub %ymm2,%ymm0,%ymm0
    1621:	62 f1 fd 28 6f 15 b5 	vmovdqa64 0x1cb5(%rip),%ymm2        # 32e0 <_IO_stdin_used+0x2e0>
    1628:	1c 00 00 
    162b:	c4 e3 7d 4c c1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm0
    1631:	c4 e2 7d 00 15 c6 1c 	vpshufb 0x1cc6(%rip),%ymm0,%ymm2        # 3300 <_IO_stdin_used+0x300>
    1638:	00 00 
    163a:	c5 fd da ca          	vpminub %ymm2,%ymm0,%ymm1
    163e:	c5 fd de c2          	vpmaxub %ymm2,%ymm0,%ymm0
    1642:	62 f1 fd 28 6f 15 d4 	vmovdqa64 0x1cd4(%rip),%ymm2        # 3320 <_IO_stdin_used+0x320>
    1649:	1c 00 00 
    164c:	c4 e3 7d 4c d1 20    	vpblendvb %ymm2,%ymm1,%ymm0,%ymm2
    1652:	62 f1 fd 28 6f 0d e4 	vmovdqa64 0x1ce4(%rip),%ymm1        # 3340 <_IO_stdin_used+0x340>
    1659:	1c 00 00 
    165c:	c4 e2 75 36 ca       	vpermd %ymm2,%ymm1,%ymm1
    1661:	c5 ed da c1          	vpminub %ymm1,%ymm2,%ymm0
    1665:	c5 ed de d1          	vpmaxub %ymm1,%ymm2,%ymm2
    1669:	62 f1 fd 28 6f 0d ed 	vmovdqa64 0x1ced(%rip),%ymm1        # 3360 <_IO_stdin_used+0x360>
    1670:	1c 00 00 
    1673:	c4 e3 6d 4c c8 10    	vpblendvb %ymm1,%ymm0,%ymm2,%ymm1
    1679:	c4 e2 75 00 05 fe 1c 	vpshufb 0x1cfe(%rip),%ymm1,%ymm0        # 3380 <_IO_stdin_used+0x380>
    1680:	00 00 
    1682:	c5 f5 da d0          	vpminub %ymm0,%ymm1,%ymm2
    1686:	c5 f5 de c8          	vpmaxub %ymm0,%ymm1,%ymm1
    168a:	62 f1 fd 28 6f 05 0c 	vmovdqa64 0x1d0c(%rip),%ymm0        # 33a0 <_IO_stdin_used+0x3a0>
    1691:	1d 00 00 
    1694:	c4 e3 75 4c c2 00    	vpblendvb %ymm0,%ymm2,%ymm1,%ymm0
    169a:	c4 e2 7d 00 0d 1d 1d 	vpshufb 0x1d1d(%rip),%ymm0,%ymm1        # 33c0 <_IO_stdin_used+0x3c0>
    16a1:	00 00 
    16a3:	c5 fd da d1          	vpminub %ymm1,%ymm0,%ymm2
    16a7:	c5 fd de c1          	vpmaxub %ymm1,%ymm0,%ymm0
    16ab:	62 f1 fd 28 6f 0d 2b 	vmovdqa64 0x1d2b(%rip),%ymm1        # 33e0 <_IO_stdin_used+0x3e0>
    16b2:	1d 00 00 
    16b5:	c4 e3 7d 4c c2 10    	vpblendvb %ymm1,%ymm2,%ymm0,%ymm0
    16bb:	62 f1 fd 28 7f 07    	vmovdqa64 %ymm0,(%rdi)
    16c1:	c5 f8 77             	vzeroupper 
    16c4:	c3                   	retq   
    16c5:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    16cc:	00 00 00 
    16cf:	90                   	nop

00000000000016d0 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_>:
    16d0:	f3 0f 1e fa          	endbr64 
    16d4:	41 56                	push   %r14
    16d6:	c5 f9 ef c0          	vpxor  %xmm0,%xmm0,%xmm0
    16da:	31 d2                	xor    %edx,%edx
    16dc:	41 55                	push   %r13
    16de:	41 54                	push   %r12
    16e0:	55                   	push   %rbp
    16e1:	48 89 f5             	mov    %rsi,%rbp
    16e4:	53                   	push   %rbx
    16e5:	48 89 fb             	mov    %rdi,%rbx
    16e8:	48 81 ec 10 02 00 00 	sub    $0x210,%rsp
    16ef:	64 48 8b 04 25 28 00 	mov    %fs:0x28,%rax
    16f6:	00 00 
    16f8:	48 89 84 24 08 02 00 	mov    %rax,0x208(%rsp)
    16ff:	00 
    1700:	31 c0                	xor    %eax,%eax
    1702:	48 8d 74 24 10       	lea    0x10(%rsp),%rsi
    1707:	c5 f8 29 04 24       	vmovaps %xmm0,(%rsp)
    170c:	89 c1                	mov    %eax,%ecx
    170e:	83 c0 20             	add    $0x20,%eax
    1711:	48 89 54 0c 10       	mov    %rdx,0x10(%rsp,%rcx,1)
    1716:	48 89 54 0c 18       	mov    %rdx,0x18(%rsp,%rcx,1)
    171b:	48 89 54 0c 20       	mov    %rdx,0x20(%rsp,%rcx,1)
    1720:	48 89 54 0c 28       	mov    %rdx,0x28(%rsp,%rcx,1)
    1725:	3d e0 01 00 00       	cmp    $0x1e0,%eax
    172a:	72 e0                	jb     170c <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_+0x3c>
    172c:	c5 fb 10 05 14 1f 00 	vmovsd 0x1f14(%rip),%xmm0        # 3648 <_ZN6timersL12units_to_strE+0x28>
    1733:	00 
    1734:	c5 fb 10 4b 48       	vmovsd 0x48(%rbx),%xmm1
    1739:	8b 53 50             	mov    0x50(%rbx),%edx
    173c:	4c 8d 0d bd 1e 00 00 	lea    0x1ebd(%rip),%r9        # 3600 <_ZN5statsL13possible_fmtsE>
    1743:	48 01 c6             	add    %rax,%rsi
    1746:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    174a:	48 8d 05 cf 1e 00 00 	lea    0x1ecf(%rip),%rax        # 3620 <_ZN6timersL12units_to_strE>
    1751:	48 8d 04 d0          	lea    (%rax,%rdx,8),%rax
    1755:	c5 fb 10 4b 40       	vmovsd 0x40(%rbx),%xmm1
    175a:	49 8d 51 08          	lea    0x8(%r9),%rdx
    175e:	49 89 d6             	mov    %rdx,%r14
    1761:	4d 0f 43 f1          	cmovae %r9,%r14
    1765:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1769:	c5 fb 10 4b 38       	vmovsd 0x38(%rbx),%xmm1
    176e:	49 89 d5             	mov    %rdx,%r13
    1771:	4d 0f 43 e9          	cmovae %r9,%r13
    1775:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1779:	c5 fb 10 4b 30       	vmovsd 0x30(%rbx),%xmm1
    177e:	49 89 d3             	mov    %rdx,%r11
    1781:	4d 0f 43 d9          	cmovae %r9,%r11
    1785:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1789:	c5 fb 10 4b 28       	vmovsd 0x28(%rbx),%xmm1
    178e:	49 89 d2             	mov    %rdx,%r10
    1791:	4d 0f 43 d1          	cmovae %r9,%r10
    1795:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1799:	c5 fb 10 4b 20       	vmovsd 0x20(%rbx),%xmm1
    179e:	49 89 d0             	mov    %rdx,%r8
    17a1:	4d 0f 43 c1          	cmovae %r9,%r8
    17a5:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17a9:	c5 fb 10 4b 18       	vmovsd 0x18(%rbx),%xmm1
    17ae:	48 89 d7             	mov    %rdx,%rdi
    17b1:	49 0f 43 f9          	cmovae %r9,%rdi
    17b5:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17b9:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    17be:	48 c7 06 00 00 00 00 	movq   $0x0,(%rsi)
    17c5:	48 c7 46 08 00 00 00 	movq   $0x0,0x8(%rsi)
    17cc:	00 
    17cd:	48 89 d6             	mov    %rdx,%rsi
    17d0:	49 0f 43 f1          	cmovae %r9,%rsi
    17d4:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17d8:	c5 fb 10 4b 08       	vmovsd 0x8(%rbx),%xmm1
    17dd:	48 89 d1             	mov    %rdx,%rcx
    17e0:	49 0f 43 c9          	cmovae %r9,%rcx
    17e4:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    17e8:	49 89 e4             	mov    %rsp,%r12
    17eb:	4c 0f 42 ca          	cmovb  %rdx,%r9
    17ef:	48 83 ec 08          	sub    $0x8,%rsp
    17f3:	50                   	push   %rax
    17f4:	ba 00 02 00 00       	mov    $0x200,%edx
    17f9:	41 56                	push   %r14
    17fb:	50                   	push   %rax
    17fc:	41 55                	push   %r13
    17fe:	50                   	push   %rax
    17ff:	41 53                	push   %r11
    1801:	50                   	push   %rax
    1802:	41 52                	push   %r10
    1804:	50                   	push   %rax
    1805:	41 50                	push   %r8
    1807:	4c 8b 03             	mov    (%rbx),%r8
    180a:	50                   	push   %rax
    180b:	57                   	push   %rdi
    180c:	4c 89 e7             	mov    %r12,%rdi
    180f:	50                   	push   %rax
    1810:	56                   	push   %rsi
    1811:	be 01 00 00 00       	mov    $0x1,%esi
    1816:	50                   	push   %rax
    1817:	51                   	push   %rcx
    1818:	48 8d 0d e9 1b 00 00 	lea    0x1be9(%rip),%rcx        # 3408 <_IO_stdin_used+0x408>
    181f:	50                   	push   %rax
    1820:	31 c0                	xor    %eax,%eax
    1822:	e8 e9 f8 ff ff       	callq  1110 <__sprintf_chk@plt>
    1827:	48 81 c4 88 00 00 00 	add    $0x88,%rsp
    182e:	ff 73 48             	pushq  0x48(%rbx)
    1831:	c5 fb 10 43 08       	vmovsd 0x8(%rbx),%xmm0
    1836:	c5 fb 10 7b 40       	vmovsd 0x40(%rbx),%xmm7
    183b:	c5 fb 10 73 38       	vmovsd 0x38(%rbx),%xmm6
    1840:	c5 fb 10 6b 30       	vmovsd 0x30(%rbx),%xmm5
    1845:	c5 fb 10 63 28       	vmovsd 0x28(%rbx),%xmm4
    184a:	c5 fb 10 5b 20       	vmovsd 0x20(%rbx),%xmm3
    184f:	c5 fb 10 53 18       	vmovsd 0x18(%rbx),%xmm2
    1854:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    1859:	4c 89 e2             	mov    %r12,%rdx
    185c:	be 01 00 00 00       	mov    $0x1,%esi
    1861:	48 89 ef             	mov    %rbp,%rdi
    1864:	b8 08 00 00 00       	mov    $0x8,%eax
    1869:	e8 82 f8 ff ff       	callq  10f0 <__fprintf_chk@plt>
    186e:	58                   	pop    %rax
    186f:	5a                   	pop    %rdx
    1870:	48 8b 84 24 08 02 00 	mov    0x208(%rsp),%rax
    1877:	00 
    1878:	64 48 33 04 25 28 00 	xor    %fs:0x28,%rax
    187f:	00 00 
    1881:	75 10                	jne    1893 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_+0x1c3>
    1883:	48 81 c4 10 02 00 00 	add    $0x210,%rsp
    188a:	5b                   	pop    %rbx
    188b:	5d                   	pop    %rbp
    188c:	41 5c                	pop    %r12
    188e:	41 5d                	pop    %r13
    1890:	41 5e                	pop    %r14
    1892:	c3                   	retq   
    1893:	e8 28 f8 ff ff       	callq  10c0 <__stack_chk_fail@plt>
    1898:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    189f:	00 

00000000000018a0 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_>:
    18a0:	f3 0f 1e fa          	endbr64 
    18a4:	41 55                	push   %r13
    18a6:	4d 89 c5             	mov    %r8,%r13
    18a9:	41 54                	push   %r12
    18ab:	49 89 d4             	mov    %rdx,%r12
    18ae:	48 89 ca             	mov    %rcx,%rdx
    18b1:	55                   	push   %rbp
    18b2:	48 89 fd             	mov    %rdi,%rbp
    18b5:	40 84 f6             	test   %sil,%sil
    18b8:	74 16                	je     18d0 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x30>
    18ba:	5d                   	pop    %rbp
    18bb:	4c 89 e6             	mov    %r12,%rsi
    18be:	41 5c                	pop    %r12
    18c0:	4c 89 c1             	mov    %r8,%rcx
    18c3:	41 5d                	pop    %r13
    18c5:	e9 06 fe ff ff       	jmpq   16d0 <_ZN5stats9stats_out8print_hrEP8_IO_FILEPKcS4_>
    18ca:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    18d0:	48 83 f9 01          	cmp    $0x1,%rcx
    18d4:	76 09                	jbe    18df <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x3f>
    18d6:	4d 85 c0             	test   %r8,%r8
    18d9:	0f 85 81 00 00 00    	jne    1960 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0xc0>
    18df:	48 83 fa 01          	cmp    $0x1,%rdx
    18e3:	0f 85 8d 00 00 00    	jne    1976 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0xd6>
    18e9:	4d 85 ed             	test   %r13,%r13
    18ec:	74 19                	je     1907 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x67>
    18ee:	4c 89 e9             	mov    %r13,%rcx
    18f1:	48 8d 15 ec 1c 00 00 	lea    0x1cec(%rip),%rdx        # 35e4 <_IO_stdin_used+0x5e4>
    18f8:	be 01 00 00 00       	mov    $0x1,%esi
    18fd:	4c 89 e7             	mov    %r12,%rdi
    1900:	31 c0                	xor    %eax,%eax
    1902:	e8 e9 f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1907:	48 83 ec 08          	sub    $0x8,%rsp
    190b:	ff 75 48             	pushq  0x48(%rbp)
    190e:	c5 fb 10 45 08       	vmovsd 0x8(%rbp),%xmm0
    1913:	c5 fb 10 7d 40       	vmovsd 0x40(%rbp),%xmm7
    1918:	c5 fb 10 75 38       	vmovsd 0x38(%rbp),%xmm6
    191d:	c5 fb 10 6d 30       	vmovsd 0x30(%rbp),%xmm5
    1922:	c5 fb 10 65 28       	vmovsd 0x28(%rbp),%xmm4
    1927:	c5 fb 10 5d 20       	vmovsd 0x20(%rbp),%xmm3
    192c:	c5 fb 10 55 18       	vmovsd 0x18(%rbp),%xmm2
    1931:	c5 fb 10 4d 10       	vmovsd 0x10(%rbp),%xmm1
    1936:	48 8b 4d 00          	mov    0x0(%rbp),%rcx
    193a:	48 8d 15 0f 1c 00 00 	lea    0x1c0f(%rip),%rdx        # 3550 <_IO_stdin_used+0x550>
    1941:	4c 89 e7             	mov    %r12,%rdi
    1944:	be 01 00 00 00       	mov    $0x1,%esi
    1949:	b8 08 00 00 00       	mov    $0x8,%eax
    194e:	e8 9d f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1953:	58                   	pop    %rax
    1954:	5a                   	pop    %rdx
    1955:	5d                   	pop    %rbp
    1956:	41 5c                	pop    %r12
    1958:	41 5d                	pop    %r13
    195a:	c3                   	retq   
    195b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1960:	48 8d 15 7d 1c 00 00 	lea    0x1c7d(%rip),%rdx        # 35e4 <_IO_stdin_used+0x5e4>
    1967:	be 01 00 00 00       	mov    $0x1,%esi
    196c:	4c 89 e7             	mov    %r12,%rdi
    196f:	31 c0                	xor    %eax,%eax
    1971:	e8 7a f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1976:	48 8d 15 9b 1b 00 00 	lea    0x1b9b(%rip),%rdx        # 3518 <_IO_stdin_used+0x518>
    197d:	be 01 00 00 00       	mov    $0x1,%esi
    1982:	4c 89 e7             	mov    %r12,%rdi
    1985:	31 c0                	xor    %eax,%eax
    1987:	e8 64 f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    198c:	e9 58 ff ff ff       	jmpq   18e9 <_ZN5stats9stats_out5printEbP8_IO_FILEPKcS4_+0x49>
    1991:	90                   	nop
    1992:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1999:	00 00 00 00 
    199d:	0f 1f 00             	nopl   (%rax)

00000000000019a0 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc>:
    19a0:	f3 0f 1e fa          	endbr64 
    19a4:	55                   	push   %rbp
    19a5:	48 89 f5             	mov    %rsi,%rbp
    19a8:	48 85 d2             	test   %rdx,%rdx
    19ab:	74 19                	je     19c6 <_ZN5stats9stats_out10export_hdrEP8_IO_FILEPKc+0x26>
    19ad:	48 89 d1             	mov    %rdx,%rcx
    19b0:	be 01 00 00 00       	mov    $0x1,%esi
    19b5:	48 8d 15 28 1c 00 00 	lea    0x1c28(%rip),%rdx        # 35e4 <_IO_stdin_used+0x5e4>
    19bc:	48 89 ef             	mov    %rbp,%rdi
    19bf:	31 c0                	xor    %eax,%eax
    19c1:	e8 2a f7 ff ff       	callq  10f0 <__fprintf_chk@plt>
    19c6:	48 89 ef             	mov    %rbp,%rdi
    19c9:	48 8d 15 48 1b 00 00 	lea    0x1b48(%rip),%rdx        # 3518 <_IO_stdin_used+0x518>
    19d0:	be 01 00 00 00       	mov    $0x1,%esi
    19d5:	31 c0                	xor    %eax,%eax
    19d7:	5d                   	pop    %rbp
    19d8:	e9 13 f7 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    19dd:	90                   	nop
    19de:	66 90                	xchg   %ax,%ax

00000000000019e0 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_>:
    19e0:	f3 0f 1e fa          	endbr64 
    19e4:	41 54                	push   %r12
    19e6:	49 89 cc             	mov    %rcx,%r12
    19e9:	55                   	push   %rbp
    19ea:	48 89 f5             	mov    %rsi,%rbp
    19ed:	53                   	push   %rbx
    19ee:	48 89 fb             	mov    %rdi,%rbx
    19f1:	48 83 fa 01          	cmp    $0x1,%rdx
    19f5:	76 09                	jbe    1a00 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x20>
    19f7:	48 85 c9             	test   %rcx,%rcx
    19fa:	0f 85 80 00 00 00    	jne    1a80 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0xa0>
    1a00:	48 83 fa 01          	cmp    $0x1,%rdx
    1a04:	0f 85 8f 00 00 00    	jne    1a99 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0xb9>
    1a0a:	4d 85 e4             	test   %r12,%r12
    1a0d:	74 19                	je     1a28 <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x48>
    1a0f:	4c 89 e1             	mov    %r12,%rcx
    1a12:	48 8d 15 cb 1b 00 00 	lea    0x1bcb(%rip),%rdx        # 35e4 <_IO_stdin_used+0x5e4>
    1a19:	be 01 00 00 00       	mov    $0x1,%esi
    1a1e:	48 89 ef             	mov    %rbp,%rdi
    1a21:	31 c0                	xor    %eax,%eax
    1a23:	e8 c8 f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1a28:	48 83 ec 08          	sub    $0x8,%rsp
    1a2c:	ff 73 48             	pushq  0x48(%rbx)
    1a2f:	c5 fb 10 43 08       	vmovsd 0x8(%rbx),%xmm0
    1a34:	c5 fb 10 7b 40       	vmovsd 0x40(%rbx),%xmm7
    1a39:	c5 fb 10 73 38       	vmovsd 0x38(%rbx),%xmm6
    1a3e:	c5 fb 10 6b 30       	vmovsd 0x30(%rbx),%xmm5
    1a43:	c5 fb 10 63 28       	vmovsd 0x28(%rbx),%xmm4
    1a48:	c5 fb 10 5b 20       	vmovsd 0x20(%rbx),%xmm3
    1a4d:	c5 fb 10 53 18       	vmovsd 0x18(%rbx),%xmm2
    1a52:	c5 fb 10 4b 10       	vmovsd 0x10(%rbx),%xmm1
    1a57:	48 8b 0b             	mov    (%rbx),%rcx
    1a5a:	48 8d 15 ef 1a 00 00 	lea    0x1aef(%rip),%rdx        # 3550 <_IO_stdin_used+0x550>
    1a61:	48 89 ef             	mov    %rbp,%rdi
    1a64:	be 01 00 00 00       	mov    $0x1,%esi
    1a69:	b8 08 00 00 00       	mov    $0x8,%eax
    1a6e:	e8 7d f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1a73:	58                   	pop    %rax
    1a74:	5a                   	pop    %rdx
    1a75:	5b                   	pop    %rbx
    1a76:	5d                   	pop    %rbp
    1a77:	41 5c                	pop    %r12
    1a79:	c3                   	retq   
    1a7a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1a80:	48 89 d1             	mov    %rdx,%rcx
    1a83:	be 01 00 00 00       	mov    $0x1,%esi
    1a88:	48 8d 15 55 1b 00 00 	lea    0x1b55(%rip),%rdx        # 35e4 <_IO_stdin_used+0x5e4>
    1a8f:	48 89 ef             	mov    %rbp,%rdi
    1a92:	31 c0                	xor    %eax,%eax
    1a94:	e8 57 f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1a99:	48 8d 15 78 1a 00 00 	lea    0x1a78(%rip),%rdx        # 3518 <_IO_stdin_used+0x518>
    1aa0:	be 01 00 00 00       	mov    $0x1,%esi
    1aa5:	48 89 ef             	mov    %rbp,%rdi
    1aa8:	31 c0                	xor    %eax,%eax
    1aaa:	e8 41 f6 ff ff       	callq  10f0 <__fprintf_chk@plt>
    1aaf:	e9 56 ff ff ff       	jmpq   1a0a <_ZN5stats9stats_out9print_csvEP8_IO_FILEPKcS4_+0x2a>
    1ab4:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1abb:	00 00 00 00 
    1abf:	90                   	nop

0000000000001ac0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE>:
    1ac0:	f3 0f 1e fa          	endbr64 
    1ac4:	89 4f 50             	mov    %ecx,0x50(%rdi)
    1ac7:	83 fa 01             	cmp    $0x1,%edx
    1aca:	0f 84 d0 01 00 00    	je     1ca0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x1e0>
    1ad0:	89 d0                	mov    %edx,%eax
    1ad2:	48 89 07             	mov    %rax,(%rdi)
    1ad5:	89 d0                	mov    %edx,%eax
    1ad7:	d1 e8                	shr    %eax
    1ad9:	89 c1                	mov    %eax,%ecx
    1adb:	c5 fb 10 04 ce       	vmovsd (%rsi,%rcx,8),%xmm0
    1ae0:	f6 c2 01             	test   $0x1,%dl
    1ae3:	0f 85 97 01 00 00    	jne    1c80 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x1c0>
    1ae9:	ff c0                	inc    %eax
    1aeb:	c5 fb 58 04 c6       	vaddsd (%rsi,%rax,8),%xmm0,%xmm0
    1af0:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
    1af7:	00 
    1af8:	c5 fb 59 05 50 1b 00 	vmulsd 0x1b50(%rip),%xmm0,%xmm0        # 3650 <_ZN6timersL12units_to_strE+0x30>
    1aff:	00 
    1b00:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1b05:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    1b09:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
    1b0d:	85 d2                	test   %edx,%edx
    1b0f:	74 20                	je     1b31 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x71>
    1b11:	31 c0                	xor    %eax,%eax
    1b13:	c5 f9 28 d0          	vmovapd %xmm0,%xmm2
    1b17:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
    1b1e:	00 00 
    1b20:	c5 eb 58 14 c6       	vaddsd (%rsi,%rax,8),%xmm2,%xmm2
    1b25:	48 ff c0             	inc    %rax
    1b28:	c5 fb 11 57 10       	vmovsd %xmm2,0x10(%rdi)
    1b2d:	39 c2                	cmp    %eax,%edx
    1b2f:	77 ef                	ja     1b20 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x60>
    1b31:	62 f1 57 08 7b ea    	vcvtusi2sd %edx,%xmm5,%xmm5
    1b37:	8d 42 ff             	lea    -0x1(%rdx),%eax
    1b3a:	48 89 c1             	mov    %rax,%rcx
    1b3d:	c5 eb 5e d5          	vdivsd %xmm5,%xmm2,%xmm2
    1b41:	c5 fb 11 57 10       	vmovsd %xmm2,0x10(%rdi)
    1b46:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1b4b:	c5 fb 11 4f 18       	vmovsd %xmm1,0x18(%rdi)
    1b50:	c5 fb 10 0e          	vmovsd (%rsi),%xmm1
    1b54:	c5 fb 11 4f 20       	vmovsd %xmm1,0x20(%rdi)
    1b59:	c5 d3 59 0d f7 1a 00 	vmulsd 0x1af7(%rip),%xmm5,%xmm1        # 3658 <_ZN6timersL12units_to_strE+0x38>
    1b60:	00 
    1b61:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1b67:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1b6c:	c5 fb 11 4f 28       	vmovsd %xmm1,0x28(%rdi)
    1b71:	c5 d3 59 0d e7 1a 00 	vmulsd 0x1ae7(%rip),%xmm5,%xmm1        # 3660 <_ZN6timersL12units_to_strE+0x40>
    1b78:	00 
    1b79:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1b7f:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1b84:	c5 fb 11 4f 30       	vmovsd %xmm1,0x30(%rdi)
    1b89:	c5 d3 59 0d d7 1a 00 	vmulsd 0x1ad7(%rip),%xmm5,%xmm1        # 3668 <_ZN6timersL12units_to_strE+0x48>
    1b90:	00 
    1b91:	62 f1 ff 08 78 c1    	vcvttsd2usi %xmm1,%rax
    1b97:	c5 fb 10 0c c6       	vmovsd (%rsi,%rax,8),%xmm1
    1b9c:	c5 fb 11 4f 38       	vmovsd %xmm1,0x38(%rdi)
    1ba1:	83 fa 01             	cmp    $0x1,%edx
    1ba4:	0f 86 56 01 00 00    	jbe    1d00 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x240>
    1baa:	83 f9 02             	cmp    $0x2,%ecx
    1bad:	0f 86 67 01 00 00    	jbe    1d1a <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x25a>
    1bb3:	41 89 d0             	mov    %edx,%r8d
    1bb6:	41 c1 e8 02          	shr    $0x2,%r8d
    1bba:	41 ff c8             	dec    %r8d
    1bbd:	49 c1 e0 05          	shl    $0x5,%r8
    1bc1:	c4 e2 7d 19 f2       	vbroadcastsd %xmm2,%ymm6
    1bc6:	48 89 f0             	mov    %rsi,%rax
    1bc9:	4e 8d 44 06 20       	lea    0x20(%rsi,%r8,1),%r8
    1bce:	c5 f9 28 d8          	vmovapd %xmm0,%xmm3
    1bd2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1bd8:	c5 fd 10 38          	vmovupd (%rax),%ymm7
    1bdc:	48 83 c0 20          	add    $0x20,%rax
    1be0:	c5 c5 5c c6          	vsubpd %ymm6,%ymm7,%ymm0
    1be4:	c5 fd 59 c0          	vmulpd %ymm0,%ymm0,%ymm0
    1be8:	c5 fb 58 db          	vaddsd %xmm3,%xmm0,%xmm3
    1bec:	c5 f9 15 c8          	vunpckhpd %xmm0,%xmm0,%xmm1
    1bf0:	62 f3 fd 28 19 c0 01 	vextractf64x2 $0x1,%ymm0,%xmm0
    1bf7:	c5 f3 58 cb          	vaddsd %xmm3,%xmm1,%xmm1
    1bfb:	c5 f3 58 c8          	vaddsd %xmm0,%xmm1,%xmm1
    1bff:	c5 f9 15 c0          	vunpckhpd %xmm0,%xmm0,%xmm0
    1c03:	c5 f3 58 d8          	vaddsd %xmm0,%xmm1,%xmm3
    1c07:	4c 39 c0             	cmp    %r8,%rax
    1c0a:	75 cc                	jne    1bd8 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x118>
    1c0c:	89 d0                	mov    %edx,%eax
    1c0e:	83 e0 fc             	and    $0xfffffffc,%eax
    1c11:	f6 c2 03             	test   $0x3,%dl
    1c14:	0f 84 d6 00 00 00    	je     1cf0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x230>
    1c1a:	41 89 c0             	mov    %eax,%r8d
    1c1d:	c4 a1 7b 10 04 c6    	vmovsd (%rsi,%r8,8),%xmm0
    1c23:	ff c0                	inc    %eax
    1c25:	c5 fb 5c c2          	vsubsd %xmm2,%xmm0,%xmm0
    1c29:	c4 e2 f9 b9 d8       	vfmadd231sd %xmm0,%xmm0,%xmm3
    1c2e:	39 c2                	cmp    %eax,%edx
    1c30:	0f 86 ba 00 00 00    	jbe    1cf0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x230>
    1c36:	c5 f8 77             	vzeroupper 
    1c39:	41 89 c0             	mov    %eax,%r8d
    1c3c:	c4 a1 7b 10 04 c6    	vmovsd (%rsi,%r8,8),%xmm0
    1c42:	ff c0                	inc    %eax
    1c44:	c5 fb 5c c2          	vsubsd %xmm2,%xmm0,%xmm0
    1c48:	c4 e2 f9 b9 d8       	vfmadd231sd %xmm0,%xmm0,%xmm3
    1c4d:	39 c2                	cmp    %eax,%edx
    1c4f:	76 0e                	jbe    1c5f <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x19f>
    1c51:	c5 fb 10 04 c6       	vmovsd (%rsi,%rax,8),%xmm0
    1c56:	c5 fb 5c d2          	vsubsd %xmm2,%xmm0,%xmm2
    1c5a:	c4 e2 e9 b9 da       	vfmadd231sd %xmm2,%xmm2,%xmm3
    1c5f:	62 f1 7f 08 7b c1    	vcvtusi2sd %ecx,%xmm0,%xmm0
    1c65:	c5 e3 5e c0          	vdivsd %xmm0,%xmm3,%xmm0
    1c69:	c5 fb 11 47 40       	vmovsd %xmm0,0x40(%rdi)
    1c6e:	c5 e3 5e c5          	vdivsd %xmm5,%xmm3,%xmm0
    1c72:	c5 fb 11 47 48       	vmovsd %xmm0,0x48(%rdi)
    1c77:	c3                   	retq   
    1c78:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1c7f:	00 
    1c80:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1c85:	48 c7 47 10 00 00 00 	movq   $0x0,0x10(%rdi)
    1c8c:	00 
    1c8d:	c5 f9 57 c0          	vxorpd %xmm0,%xmm0,%xmm0
    1c91:	e9 7b fe ff ff       	jmpq   1b11 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x51>
    1c96:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1c9d:	00 00 00 
    1ca0:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1ca4:	48 c7 07 01 00 00 00 	movq   $0x1,(%rdi)
    1cab:	c5 fb 11 47 08       	vmovsd %xmm0,0x8(%rdi)
    1cb0:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1cb4:	c5 fb 11 47 10       	vmovsd %xmm0,0x10(%rdi)
    1cb9:	c5 fb 11 47 18       	vmovsd %xmm0,0x18(%rdi)
    1cbe:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1cc2:	c5 fb 11 47 20       	vmovsd %xmm0,0x20(%rdi)
    1cc7:	c5 fb 11 47 28       	vmovsd %xmm0,0x28(%rdi)
    1ccc:	c5 fb 10 06          	vmovsd (%rsi),%xmm0
    1cd0:	48 c7 47 40 00 00 00 	movq   $0x0,0x40(%rdi)
    1cd7:	00 
    1cd8:	48 c7 47 48 00 00 00 	movq   $0x0,0x48(%rdi)
    1cdf:	00 
    1ce0:	c5 fb 11 47 30       	vmovsd %xmm0,0x30(%rdi)
    1ce5:	c5 fb 11 47 38       	vmovsd %xmm0,0x38(%rdi)
    1cea:	c3                   	retq   
    1ceb:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1cf0:	c5 f8 77             	vzeroupper 
    1cf3:	e9 67 ff ff ff       	jmpq   1c5f <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x19f>
    1cf8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    1cff:	00 
    1d00:	48 8b 3d 19 33 00 00 	mov    0x3319(%rip),%rdi        # 5020 <stderr@@GLIBC_2.2.5>
    1d07:	48 8d 15 72 18 00 00 	lea    0x1872(%rip),%rdx        # 3580 <_IO_stdin_used+0x580>
    1d0e:	be 01 00 00 00       	mov    $0x1,%esi
    1d13:	31 c0                	xor    %eax,%eax
    1d15:	e9 d6 f3 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    1d1a:	c5 fb 10 0e          	vmovsd (%rsi),%xmm1
    1d1e:	b8 01 00 00 00       	mov    $0x1,%eax
    1d23:	c5 f3 5c ca          	vsubsd %xmm2,%xmm1,%xmm1
    1d27:	c5 f9 28 d9          	vmovapd %xmm1,%xmm3
    1d2b:	c4 e2 f9 99 d9       	vfmadd132sd %xmm1,%xmm0,%xmm3
    1d30:	e9 04 ff ff ff       	jmpq   1c39 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE+0x179>
    1d35:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    1d3c:	00 00 00 00 

0000000000001d40 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>:
    1d40:	48 89 f0             	mov    %rsi,%rax
    1d43:	48 29 f8             	sub    %rdi,%rax
    1d46:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    1d4c:	0f 8e a6 01 00 00    	jle    1ef8 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x1b8>
    1d52:	41 56                	push   %r14
    1d54:	49 89 d6             	mov    %rdx,%r14
    1d57:	41 55                	push   %r13
    1d59:	4c 8d 6f 08          	lea    0x8(%rdi),%r13
    1d5d:	41 54                	push   %r12
    1d5f:	55                   	push   %rbp
    1d60:	53                   	push   %rbx
    1d61:	48 89 fb             	mov    %rdi,%rbx
    1d64:	48 85 d2             	test   %rdx,%rdx
    1d67:	0f 84 03 01 00 00    	je     1e70 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x130>
    1d6d:	48 c1 f8 04          	sar    $0x4,%rax
    1d71:	48 8d 04 c3          	lea    (%rbx,%rax,8),%rax
    1d75:	c5 fb 10 4b 08       	vmovsd 0x8(%rbx),%xmm1
    1d7a:	c5 fb 10 10          	vmovsd (%rax),%xmm2
    1d7e:	49 ff ce             	dec    %r14
    1d81:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
    1d85:	c5 fb 10 5e f8       	vmovsd -0x8(%rsi),%xmm3
    1d8a:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    1d8e:	76 70                	jbe    1e00 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xc0>
    1d90:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
    1d94:	0f 87 b6 00 00 00    	ja     1e50 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x110>
    1d9a:	c5 f9 2f d9          	vcomisd %xmm1,%xmm3
    1d9e:	77 6c                	ja     1e0c <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xcc>
    1da0:	c5 f1 14 c0          	vunpcklpd %xmm0,%xmm1,%xmm0
    1da4:	c5 f8 11 03          	vmovups %xmm0,(%rbx)
    1da8:	c5 fb 10 46 f8       	vmovsd -0x8(%rsi),%xmm0
    1dad:	4c 89 ed             	mov    %r13,%rbp
    1db0:	48 89 f0             	mov    %rsi,%rax
    1db3:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1db8:	c5 fb 10 55 00       	vmovsd 0x0(%rbp),%xmm2
    1dbd:	49 89 ec             	mov    %rbp,%r12
    1dc0:	c5 f9 2f ca          	vcomisd %xmm2,%xmm1
    1dc4:	77 30                	ja     1df6 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xb6>
    1dc6:	48 83 e8 08          	sub    $0x8,%rax
    1dca:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1dce:	76 0f                	jbe    1ddf <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x9f>
    1dd0:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1dd5:	48 83 e8 08          	sub    $0x8,%rax
    1dd9:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    1ddd:	77 f1                	ja     1dd0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x90>
    1ddf:	48 39 e8             	cmp    %rbp,%rax
    1de2:	76 3c                	jbe    1e20 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0xe0>
    1de4:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    1de9:	c5 fb 11 10          	vmovsd %xmm2,(%rax)
    1ded:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1df1:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    1df6:	48 83 c5 08          	add    $0x8,%rbp
    1dfa:	eb bc                	jmp    1db8 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x78>
    1dfc:	0f 1f 40 00          	nopl   0x0(%rax)
    1e00:	c5 f9 2f d9          	vcomisd %xmm1,%xmm3
    1e04:	77 9a                	ja     1da0 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x60>
    1e06:	c5 f9 2f da          	vcomisd %xmm2,%xmm3
    1e0a:	76 44                	jbe    1e50 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x110>
    1e0c:	c5 fb 11 1b          	vmovsd %xmm3,(%rbx)
    1e10:	c5 fb 11 46 f8       	vmovsd %xmm0,-0x8(%rsi)
    1e15:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1e19:	eb 92                	jmp    1dad <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x6d>
    1e1b:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    1e20:	4c 89 f2             	mov    %r14,%rdx
    1e23:	48 89 ef             	mov    %rbp,%rdi
    1e26:	e8 15 ff ff ff       	callq  1d40 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>
    1e2b:	48 89 e8             	mov    %rbp,%rax
    1e2e:	48 29 d8             	sub    %rbx,%rax
    1e31:	48 3d 80 00 00 00    	cmp    $0x80,%rax
    1e37:	0f 8e ac 00 00 00    	jle    1ee9 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x1a9>
    1e3d:	4d 85 f6             	test   %r14,%r14
    1e40:	74 31                	je     1e73 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x133>
    1e42:	48 89 ee             	mov    %rbp,%rsi
    1e45:	e9 23 ff ff ff       	jmpq   1d6d <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x2d>
    1e4a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1e50:	c5 fb 11 13          	vmovsd %xmm2,(%rbx)
    1e54:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    1e58:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1e5c:	c5 fb 10 46 f8       	vmovsd -0x8(%rsi),%xmm0
    1e61:	e9 47 ff ff ff       	jmpq   1dad <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x6d>
    1e66:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    1e6d:	00 00 00 
    1e70:	49 89 f4             	mov    %rsi,%r12
    1e73:	48 c1 f8 03          	sar    $0x3,%rax
    1e77:	4c 8d 68 fe          	lea    -0x2(%rax),%r13
    1e7b:	49 d1 fd             	sar    %r13
    1e7e:	c4 a1 7b 10 04 eb    	vmovsd (%rbx,%r13,8),%xmm0
    1e84:	48 89 c2             	mov    %rax,%rdx
    1e87:	4c 89 ee             	mov    %r13,%rsi
    1e8a:	48 89 df             	mov    %rbx,%rdi
    1e8d:	48 89 c5             	mov    %rax,%rbp
    1e90:	e8 6b 04 00 00       	callq  2300 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1e95:	49 ff cd             	dec    %r13
    1e98:	c4 a1 7b 10 04 eb    	vmovsd (%rbx,%r13,8),%xmm0
    1e9e:	48 89 ea             	mov    %rbp,%rdx
    1ea1:	4c 89 ee             	mov    %r13,%rsi
    1ea4:	48 89 df             	mov    %rbx,%rdi
    1ea7:	e8 54 04 00 00       	callq  2300 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1eac:	4d 85 ed             	test   %r13,%r13
    1eaf:	75 e4                	jne    1e95 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x155>
    1eb1:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1eb8:	49 83 ec 08          	sub    $0x8,%r12
    1ebc:	4c 89 e5             	mov    %r12,%rbp
    1ebf:	48 29 dd             	sub    %rbx,%rbp
    1ec2:	c5 fb 10 0b          	vmovsd (%rbx),%xmm1
    1ec6:	48 89 ea             	mov    %rbp,%rdx
    1ec9:	c4 c1 7b 10 04 24    	vmovsd (%r12),%xmm0
    1ecf:	48 c1 fa 03          	sar    $0x3,%rdx
    1ed3:	31 f6                	xor    %esi,%esi
    1ed5:	c4 c1 7b 11 0c 24    	vmovsd %xmm1,(%r12)
    1edb:	48 89 df             	mov    %rbx,%rdi
    1ede:	e8 1d 04 00 00       	callq  2300 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>
    1ee3:	48 83 fd 08          	cmp    $0x8,%rbp
    1ee7:	7f cf                	jg     1eb8 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0+0x178>
    1ee9:	5b                   	pop    %rbx
    1eea:	5d                   	pop    %rbp
    1eeb:	41 5c                	pop    %r12
    1eed:	41 5d                	pop    %r13
    1eef:	41 5e                	pop    %r14
    1ef1:	c3                   	retq   
    1ef2:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    1ef8:	c3                   	retq   
    1ef9:	90                   	nop
    1efa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)

0000000000001f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>:
    1f00:	f3 0f 1e fa          	endbr64 
    1f04:	41 57                	push   %r15
    1f06:	41 56                	push   %r14
    1f08:	41 55                	push   %r13
    1f0a:	41 54                	push   %r12
    1f0c:	55                   	push   %rbp
    1f0d:	53                   	push   %rbx
    1f0e:	48 83 ec 18          	sub    $0x18,%rsp
    1f12:	48 89 3c 24          	mov    %rdi,(%rsp)
    1f16:	85 d2                	test   %edx,%edx
    1f18:	0f 84 5a 01 00 00    	je     2078 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x178>
    1f1e:	48 89 f5             	mov    %rsi,%rbp
    1f21:	48 85 f6             	test   %rsi,%rsi
    1f24:	0f 84 4e 01 00 00    	je     2078 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x178>
    1f2a:	41 89 d4             	mov    %edx,%r12d
    1f2d:	49 c1 e4 03          	shl    $0x3,%r12
    1f31:	4c 89 e0             	mov    %r12,%rax
    1f34:	41 89 d7             	mov    %edx,%r15d
    1f37:	48 c1 f8 03          	sar    $0x3,%rax
    1f3b:	ba 3f 00 00 00       	mov    $0x3f,%edx
    1f40:	f3 48 0f bd c0       	lzcnt  %rax,%rax
    1f45:	29 c2                	sub    %eax,%edx
    1f47:	4e 8d 2c 26          	lea    (%rsi,%r12,1),%r13
    1f4b:	48 63 d2             	movslq %edx,%rdx
    1f4e:	48 01 d2             	add    %rdx,%rdx
    1f51:	4c 89 ee             	mov    %r13,%rsi
    1f54:	48 89 ef             	mov    %rbp,%rdi
    1f57:	41 89 ce             	mov    %ecx,%r14d
    1f5a:	48 8d 5d 08          	lea    0x8(%rbp),%rbx
    1f5e:	e8 dd fd ff ff       	callq  1d40 <_ZSt16__introsort_loopIPdlN9__gnu_cxx5__ops15_Iter_less_iterEEvT_S4_T0_T1_.isra.0>
    1f63:	49 81 fc 80 00 00 00 	cmp    $0x80,%r12
    1f6a:	0f 86 30 01 00 00    	jbe    20a0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1a0>
    1f70:	4c 8d a5 80 00 00 00 	lea    0x80(%rbp),%r12
    1f77:	eb 3e                	jmp    1fb7 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xb7>
    1f79:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1f80:	48 39 dd             	cmp    %rbx,%rbp
    1f83:	74 24                	je     1fa9 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xa9>
    1f85:	b8 08 00 00 00       	mov    $0x8,%eax
    1f8a:	48 89 da             	mov    %rbx,%rdx
    1f8d:	48 29 ea             	sub    %rbp,%rdx
    1f90:	48 8d 7c 05 00       	lea    0x0(%rbp,%rax,1),%rdi
    1f95:	48 89 ee             	mov    %rbp,%rsi
    1f98:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    1f9e:	e8 3d f1 ff ff       	callq  10e0 <memmove@plt>
    1fa3:	c5 fb 10 44 24 08    	vmovsd 0x8(%rsp),%xmm0
    1fa9:	48 83 c3 08          	add    $0x8,%rbx
    1fad:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    1fb2:	49 39 dc             	cmp    %rbx,%r12
    1fb5:	74 4d                	je     2004 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x104>
    1fb7:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    1fbb:	c5 fb 10 4d 00       	vmovsd 0x0(%rbp),%xmm1
    1fc0:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1fc4:	77 ba                	ja     1f80 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x80>
    1fc6:	c5 fb 10 4b f8       	vmovsd -0x8(%rbx),%xmm1
    1fcb:	48 8d 43 f8          	lea    -0x8(%rbx),%rax
    1fcf:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1fd3:	0f 86 67 01 00 00    	jbe    2140 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x240>
    1fd9:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    1fe0:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
    1fe5:	48 89 c2             	mov    %rax,%rdx
    1fe8:	c5 fb 10 48 f8       	vmovsd -0x8(%rax),%xmm1
    1fed:	48 83 e8 08          	sub    $0x8,%rax
    1ff1:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    1ff5:	77 e9                	ja     1fe0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xe0>
    1ff7:	48 83 c3 08          	add    $0x8,%rbx
    1ffb:	c5 fb 11 02          	vmovsd %xmm0,(%rdx)
    1fff:	49 39 dc             	cmp    %rbx,%r12
    2002:	75 b3                	jne    1fb7 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xb7>
    2004:	4d 39 e5             	cmp    %r12,%r13
    2007:	74 4b                	je     2054 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    2009:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2010:	c4 c1 7b 10 0c 24    	vmovsd (%r12),%xmm1
    2016:	c4 c1 7b 10 44 24 f8 	vmovsd -0x8(%r12),%xmm0
    201d:	49 8d 44 24 f8       	lea    -0x8(%r12),%rax
    2022:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    2026:	0f 86 fc 00 00 00    	jbe    2128 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x228>
    202c:	0f 1f 40 00          	nopl   0x0(%rax)
    2030:	c5 fb 11 40 08       	vmovsd %xmm0,0x8(%rax)
    2035:	48 89 c2             	mov    %rax,%rdx
    2038:	c5 fb 10 40 f8       	vmovsd -0x8(%rax),%xmm0
    203d:	48 83 e8 08          	sub    $0x8,%rax
    2041:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    2045:	77 e9                	ja     2030 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x130>
    2047:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
    204b:	49 83 c4 08          	add    $0x8,%r12
    204f:	4d 39 e5             	cmp    %r12,%r13
    2052:	75 bc                	jne    2010 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x110>
    2054:	48 8b 3c 24          	mov    (%rsp),%rdi
    2058:	48 83 c4 18          	add    $0x18,%rsp
    205c:	5b                   	pop    %rbx
    205d:	48 89 ee             	mov    %rbp,%rsi
    2060:	5d                   	pop    %rbp
    2061:	41 5c                	pop    %r12
    2063:	41 5d                	pop    %r13
    2065:	44 89 f1             	mov    %r14d,%ecx
    2068:	41 5e                	pop    %r14
    206a:	44 89 fa             	mov    %r15d,%edx
    206d:	41 5f                	pop    %r15
    206f:	e9 4c fa ff ff       	jmpq   1ac0 <_ZN5stats9stats_out21sorted_array_to_statsEPdjN6timers10time_unitsE>
    2074:	0f 1f 40 00          	nopl   0x0(%rax)
    2078:	48 8b 3d a1 2f 00 00 	mov    0x2fa1(%rip),%rdi        # 5020 <stderr@@GLIBC_2.2.5>
    207f:	48 83 c4 18          	add    $0x18,%rsp
    2083:	5b                   	pop    %rbx
    2084:	5d                   	pop    %rbp
    2085:	41 5c                	pop    %r12
    2087:	41 5d                	pop    %r13
    2089:	41 5e                	pop    %r14
    208b:	48 8d 15 2e 15 00 00 	lea    0x152e(%rip),%rdx        # 35c0 <_IO_stdin_used+0x5c0>
    2092:	be 01 00 00 00       	mov    $0x1,%esi
    2097:	31 c0                	xor    %eax,%eax
    2099:	41 5f                	pop    %r15
    209b:	e9 50 f0 ff ff       	jmpq   10f0 <__fprintf_chk@plt>
    20a0:	41 bc 08 00 00 00    	mov    $0x8,%r12d
    20a6:	49 39 dd             	cmp    %rbx,%r13
    20a9:	75 3b                	jne    20e6 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1e6>
    20ab:	eb a7                	jmp    2054 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    20ad:	0f 1f 00             	nopl   (%rax)
    20b0:	48 39 dd             	cmp    %rbx,%rbp
    20b3:	74 1f                	je     20d4 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1d4>
    20b5:	48 89 da             	mov    %rbx,%rdx
    20b8:	48 29 ea             	sub    %rbp,%rdx
    20bb:	4a 8d 7c 25 00       	lea    0x0(%rbp,%r12,1),%rdi
    20c0:	48 89 ee             	mov    %rbp,%rsi
    20c3:	c5 fb 11 44 24 08    	vmovsd %xmm0,0x8(%rsp)
    20c9:	e8 12 f0 ff ff       	callq  10e0 <memmove@plt>
    20ce:	c5 fb 10 44 24 08    	vmovsd 0x8(%rsp),%xmm0
    20d4:	c5 fb 11 45 00       	vmovsd %xmm0,0x0(%rbp)
    20d9:	48 83 c3 08          	add    $0x8,%rbx
    20dd:	49 39 dd             	cmp    %rbx,%r13
    20e0:	0f 84 6e ff ff ff    	je     2054 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x154>
    20e6:	c5 fb 10 03          	vmovsd (%rbx),%xmm0
    20ea:	c5 fb 10 4d 00       	vmovsd 0x0(%rbp),%xmm1
    20ef:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    20f3:	77 bb                	ja     20b0 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1b0>
    20f5:	c5 fb 10 4b f8       	vmovsd -0x8(%rbx),%xmm1
    20fa:	48 8d 43 f8          	lea    -0x8(%rbx),%rax
    20fe:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    2102:	76 4c                	jbe    2150 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x250>
    2104:	0f 1f 40 00          	nopl   0x0(%rax)
    2108:	c5 fb 11 48 08       	vmovsd %xmm1,0x8(%rax)
    210d:	48 89 c2             	mov    %rax,%rdx
    2110:	c5 fb 10 48 f8       	vmovsd -0x8(%rax),%xmm1
    2115:	48 83 e8 08          	sub    $0x8,%rax
    2119:	c5 f9 2f c8          	vcomisd %xmm0,%xmm1
    211d:	77 e9                	ja     2108 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x208>
    211f:	c5 fb 11 02          	vmovsd %xmm0,(%rdx)
    2123:	eb b4                	jmp    20d9 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x1d9>
    2125:	0f 1f 00             	nopl   (%rax)
    2128:	4c 89 e2             	mov    %r12,%rdx
    212b:	c5 fb 11 0a          	vmovsd %xmm1,(%rdx)
    212f:	49 83 c4 08          	add    $0x8,%r12
    2133:	e9 17 ff ff ff       	jmpq   204f <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x14f>
    2138:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    213f:	00 
    2140:	48 89 da             	mov    %rbx,%rdx
    2143:	e9 af fe ff ff       	jmpq   1ff7 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0xf7>
    2148:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    214f:	00 
    2150:	48 89 da             	mov    %rbx,%rdx
    2153:	eb ca                	jmp    211f <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE+0x21f>
    2155:	90                   	nop
    2156:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    215d:	00 00 00 

0000000000002160 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE>:
    2160:	f3 0f 1e fa          	endbr64 
    2164:	85 d2                	test   %edx,%edx
    2166:	0f 84 84 00 00 00    	je     21f0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    216c:	8d 42 ff             	lea    -0x1(%rdx),%eax
    216f:	83 f8 02             	cmp    $0x2,%eax
    2172:	0f 86 88 00 00 00    	jbe    2200 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0xa0>
    2178:	41 89 d0             	mov    %edx,%r8d
    217b:	41 c1 e8 02          	shr    $0x2,%r8d
    217f:	41 ff c8             	dec    %r8d
    2182:	49 c1 e0 05          	shl    $0x5,%r8
    2186:	48 89 f0             	mov    %rsi,%rax
    2189:	4e 8d 44 06 20       	lea    0x20(%rsi,%r8,1),%r8
    218e:	66 90                	xchg   %ax,%ax
    2190:	62 f1 fe 28 7a 00    	vcvtuqq2pd (%rax),%ymm0
    2196:	48 83 c0 20          	add    $0x20,%rax
    219a:	c5 fd 11 40 e0       	vmovupd %ymm0,-0x20(%rax)
    219f:	4c 39 c0             	cmp    %r8,%rax
    21a2:	75 ec                	jne    2190 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x30>
    21a4:	89 d0                	mov    %edx,%eax
    21a6:	83 e0 fc             	and    $0xfffffffc,%eax
    21a9:	f6 c2 03             	test   $0x3,%dl
    21ac:	74 4a                	je     21f8 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x98>
    21ae:	c5 f8 77             	vzeroupper 
    21b1:	41 89 c0             	mov    %eax,%r8d
    21b4:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    21b8:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    21be:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    21c3:	44 8d 40 01          	lea    0x1(%rax),%r8d
    21c7:	44 39 c2             	cmp    %r8d,%edx
    21ca:	76 24                	jbe    21f0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    21cc:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    21d0:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    21d6:	83 c0 02             	add    $0x2,%eax
    21d9:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    21de:	39 c2                	cmp    %eax,%edx
    21e0:	76 0e                	jbe    21f0 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x90>
    21e2:	48 8d 04 c6          	lea    (%rsi,%rax,8),%rax
    21e6:	62 f1 ff 08 7b 00    	vcvtusi2sdq (%rax),%xmm0,%xmm0
    21ec:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    21f0:	e9 0b fd ff ff       	jmpq   1f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    21f5:	0f 1f 00             	nopl   (%rax)
    21f8:	c5 f8 77             	vzeroupper 
    21fb:	e9 00 fd ff ff       	jmpq   1f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    2200:	31 c0                	xor    %eax,%eax
    2202:	eb ad                	jmp    21b1 <_ZN5stats9stats_out9get_statsEPmjN6timers10time_unitsE+0x51>
    2204:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    220b:	00 00 00 00 
    220f:	90                   	nop

0000000000002210 <_ZN5stats9stats_out9get_statsERSt6vectorIdSaIdEEN6timers10time_unitsE>:
    2210:	f3 0f 1e fa          	endbr64 
    2214:	4c 8b 06             	mov    (%rsi),%r8
    2217:	89 d1                	mov    %edx,%ecx
    2219:	48 8b 56 08          	mov    0x8(%rsi),%rdx
    221d:	4c 89 c6             	mov    %r8,%rsi
    2220:	4c 29 c2             	sub    %r8,%rdx
    2223:	48 c1 fa 03          	sar    $0x3,%rdx
    2227:	e9 d4 fc ff ff       	jmpq   1f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    222c:	0f 1f 40 00          	nopl   0x0(%rax)

0000000000002230 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE>:
    2230:	f3 0f 1e fa          	endbr64 
    2234:	48 89 f0             	mov    %rsi,%rax
    2237:	89 d1                	mov    %edx,%ecx
    2239:	48 8b 36             	mov    (%rsi),%rsi
    223c:	48 8b 50 08          	mov    0x8(%rax),%rdx
    2240:	48 29 f2             	sub    %rsi,%rdx
    2243:	48 c1 fa 03          	sar    $0x3,%rdx
    2247:	85 d2                	test   %edx,%edx
    2249:	0f 84 82 00 00 00    	je     22d1 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    224f:	8d 42 ff             	lea    -0x1(%rdx),%eax
    2252:	83 f8 02             	cmp    $0x2,%eax
    2255:	0f 86 95 00 00 00    	jbe    22f0 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xc0>
    225b:	89 d0                	mov    %edx,%eax
    225d:	c1 e8 02             	shr    $0x2,%eax
    2260:	48 c1 e0 05          	shl    $0x5,%rax
    2264:	49 89 f0             	mov    %rsi,%r8
    2267:	48 01 f0             	add    %rsi,%rax
    226a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2270:	62 d1 fe 28 7a 00    	vcvtuqq2pd (%r8),%ymm0
    2276:	49 83 c0 20          	add    $0x20,%r8
    227a:	c4 c1 7d 11 40 e0    	vmovupd %ymm0,-0x20(%r8)
    2280:	49 39 c0             	cmp    %rax,%r8
    2283:	75 eb                	jne    2270 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0x40>
    2285:	89 d0                	mov    %edx,%eax
    2287:	83 e0 fc             	and    $0xfffffffc,%eax
    228a:	f6 c2 03             	test   $0x3,%dl
    228d:	74 51                	je     22e0 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xb0>
    228f:	c5 f8 77             	vzeroupper 
    2292:	41 89 c0             	mov    %eax,%r8d
    2295:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    2299:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    229f:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    22a4:	44 8d 40 01          	lea    0x1(%rax),%r8d
    22a8:	44 39 c2             	cmp    %r8d,%edx
    22ab:	76 24                	jbe    22d1 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    22ad:	4e 8d 04 c6          	lea    (%rsi,%r8,8),%r8
    22b1:	62 d1 ff 08 7b 00    	vcvtusi2sdq (%r8),%xmm0,%xmm0
    22b7:	83 c0 02             	add    $0x2,%eax
    22ba:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    22bf:	39 c2                	cmp    %eax,%edx
    22c1:	76 0e                	jbe    22d1 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0xa1>
    22c3:	48 8d 04 c6          	lea    (%rsi,%rax,8),%rax
    22c7:	62 f1 ff 08 7b 00    	vcvtusi2sdq (%rax),%xmm0,%xmm0
    22cd:	c5 fb 11 00          	vmovsd %xmm0,(%rax)
    22d1:	e9 2a fc ff ff       	jmpq   1f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    22d6:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    22dd:	00 00 00 
    22e0:	c5 f8 77             	vzeroupper 
    22e3:	e9 18 fc ff ff       	jmpq   1f00 <_ZN5stats9stats_out9get_statsEPdjN6timers10time_unitsE>
    22e8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
    22ef:	00 
    22f0:	31 c0                	xor    %eax,%eax
    22f2:	eb 9e                	jmp    2292 <_ZN5stats9stats_out9get_statsERSt6vectorImSaImEEN6timers10time_unitsE+0x62>
    22f4:	66 2e 0f 1f 84 00 00 	nopw   %cs:0x0(%rax,%rax,1)
    22fb:	00 00 00 
    22fe:	66 90                	xchg   %ax,%ax

0000000000002300 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_>:
    2300:	f3 0f 1e fa          	endbr64 
    2304:	48 8d 42 ff          	lea    -0x1(%rdx),%rax
    2308:	49 89 c2             	mov    %rax,%r10
    230b:	49 c1 ea 3f          	shr    $0x3f,%r10
    230f:	49 01 c2             	add    %rax,%r10
    2312:	49 89 d3             	mov    %rdx,%r11
    2315:	49 d1 fa             	sar    %r10
    2318:	41 83 e3 01          	and    $0x1,%r11d
    231c:	4c 39 d6             	cmp    %r10,%rsi
    231f:	0f 8d b3 00 00 00    	jge    23d8 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xd8>
    2325:	49 89 f1             	mov    %rsi,%r9
    2328:	eb 20                	jmp    234a <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x4a>
    232a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
    2330:	48 ff c8             	dec    %rax
    2333:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    2337:	c4 c1 7b 10 08       	vmovsd (%r8),%xmm1
    233c:	c4 a1 7b 11 0c cf    	vmovsd %xmm1,(%rdi,%r9,8)
    2342:	49 39 c2             	cmp    %rax,%r10
    2345:	7e 2f                	jle    2376 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x76>
    2347:	49 89 c1             	mov    %rax,%r9
    234a:	49 8d 49 01          	lea    0x1(%r9),%rcx
    234e:	48 8d 04 09          	lea    (%rcx,%rcx,1),%rax
    2352:	48 c1 e1 04          	shl    $0x4,%rcx
    2356:	4c 8d 04 0f          	lea    (%rdi,%rcx,1),%r8
    235a:	c4 c1 7b 10 08       	vmovsd (%r8),%xmm1
    235f:	c5 fb 10 54 0f f8    	vmovsd -0x8(%rdi,%rcx,1),%xmm2
    2365:	c5 f9 2f d1          	vcomisd %xmm1,%xmm2
    2369:	77 c5                	ja     2330 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x30>
    236b:	c4 a1 7b 11 0c cf    	vmovsd %xmm1,(%rdi,%r9,8)
    2371:	49 39 c2             	cmp    %rax,%r10
    2374:	7f d1                	jg     2347 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x47>
    2376:	4d 85 db             	test   %r11,%r11
    2379:	74 6d                	je     23e8 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xe8>
    237b:	48 8d 50 ff          	lea    -0x1(%rax),%rdx
    237f:	48 89 d1             	mov    %rdx,%rcx
    2382:	48 c1 e9 3f          	shr    $0x3f,%rcx
    2386:	48 01 d1             	add    %rdx,%rcx
    2389:	48 d1 f9             	sar    %rcx
    238c:	48 39 f0             	cmp    %rsi,%rax
    238f:	7f 2b                	jg     23bc <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xbc>
    2391:	eb 3c                	jmp    23cf <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xcf>
    2393:	0f 1f 44 00 00       	nopl   0x0(%rax,%rax,1)
    2398:	48 8d 51 ff          	lea    -0x1(%rcx),%rdx
    239c:	48 89 d0             	mov    %rdx,%rax
    239f:	48 c1 e8 3f          	shr    $0x3f,%rax
    23a3:	48 01 d0             	add    %rdx,%rax
    23a6:	48 d1 f8             	sar    %rax
    23a9:	48 89 c2             	mov    %rax,%rdx
    23ac:	c4 c1 7b 11 08       	vmovsd %xmm1,(%r8)
    23b1:	48 89 c8             	mov    %rcx,%rax
    23b4:	48 39 ce             	cmp    %rcx,%rsi
    23b7:	7d 67                	jge    2420 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x120>
    23b9:	48 89 d1             	mov    %rdx,%rcx
    23bc:	4c 8d 0c cf          	lea    (%rdi,%rcx,8),%r9
    23c0:	c4 c1 7b 10 09       	vmovsd (%r9),%xmm1
    23c5:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    23c9:	c5 f9 2f c1          	vcomisd %xmm1,%xmm0
    23cd:	77 c9                	ja     2398 <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x98>
    23cf:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    23d4:	c3                   	retq   
    23d5:	0f 1f 00             	nopl   (%rax)
    23d8:	4c 8d 04 f7          	lea    (%rdi,%rsi,8),%r8
    23dc:	4d 85 db             	test   %r11,%r11
    23df:	75 ee                	jne    23cf <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0xcf>
    23e1:	48 89 f0             	mov    %rsi,%rax
    23e4:	0f 1f 40 00          	nopl   0x0(%rax)
    23e8:	48 83 ea 02          	sub    $0x2,%rdx
    23ec:	48 89 d1             	mov    %rdx,%rcx
    23ef:	48 c1 e9 3f          	shr    $0x3f,%rcx
    23f3:	48 01 ca             	add    %rcx,%rdx
    23f6:	48 d1 fa             	sar    %rdx
    23f9:	48 39 c2             	cmp    %rax,%rdx
    23fc:	0f 85 79 ff ff ff    	jne    237b <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x7b>
    2402:	48 8d 44 00 02       	lea    0x2(%rax,%rax,1),%rax
    2407:	c5 fb 10 4c c7 f8    	vmovsd -0x8(%rdi,%rax,8),%xmm1
    240d:	48 ff c8             	dec    %rax
    2410:	c4 c1 7b 11 08       	vmovsd %xmm1,(%r8)
    2415:	4c 8d 04 c7          	lea    (%rdi,%rax,8),%r8
    2419:	e9 5d ff ff ff       	jmpq   237b <_ZSt13__adjust_heapIPdldN9__gnu_cxx5__ops15_Iter_less_iterEEvT_T0_S5_T1_T2_+0x7b>
    241e:	66 90                	xchg   %ax,%ax
    2420:	4d 89 c8             	mov    %r9,%r8
    2423:	c4 c1 7b 11 00       	vmovsd %xmm0,(%r8)
    2428:	c3                   	retq   
    2429:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)

0000000000002430 <__libc_csu_init>:
    2430:	f3 0f 1e fa          	endbr64 
    2434:	41 57                	push   %r15
    2436:	4c 8d 3d 4b 29 00 00 	lea    0x294b(%rip),%r15        # 4d88 <__frame_dummy_init_array_entry>
    243d:	41 56                	push   %r14
    243f:	49 89 d6             	mov    %rdx,%r14
    2442:	41 55                	push   %r13
    2444:	49 89 f5             	mov    %rsi,%r13
    2447:	41 54                	push   %r12
    2449:	41 89 fc             	mov    %edi,%r12d
    244c:	55                   	push   %rbp
    244d:	48 8d 2d 3c 29 00 00 	lea    0x293c(%rip),%rbp        # 4d90 <__do_global_dtors_aux_fini_array_entry>
    2454:	53                   	push   %rbx
    2455:	4c 29 fd             	sub    %r15,%rbp
    2458:	48 83 ec 08          	sub    $0x8,%rsp
    245c:	e8 9f eb ff ff       	callq  1000 <_init>
    2461:	48 c1 fd 03          	sar    $0x3,%rbp
    2465:	74 1f                	je     2486 <__libc_csu_init+0x56>
    2467:	31 db                	xor    %ebx,%ebx
    2469:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
    2470:	4c 89 f2             	mov    %r14,%rdx
    2473:	4c 89 ee             	mov    %r13,%rsi
    2476:	44 89 e7             	mov    %r12d,%edi
    2479:	41 ff 14 df          	callq  *(%r15,%rbx,8)
    247d:	48 83 c3 01          	add    $0x1,%rbx
    2481:	48 39 dd             	cmp    %rbx,%rbp
    2484:	75 ea                	jne    2470 <__libc_csu_init+0x40>
    2486:	48 83 c4 08          	add    $0x8,%rsp
    248a:	5b                   	pop    %rbx
    248b:	5d                   	pop    %rbp
    248c:	41 5c                	pop    %r12
    248e:	41 5d                	pop    %r13
    2490:	41 5e                	pop    %r14
    2492:	41 5f                	pop    %r15
    2494:	c3                   	retq   
    2495:	66 66 2e 0f 1f 84 00 	data16 nopw %cs:0x0(%rax,%rax,1)
    249c:	00 00 00 00 

00000000000024a0 <__libc_csu_fini>:
    24a0:	f3 0f 1e fa          	endbr64 
    24a4:	c3                   	retq   

Disassembly of section .fini:

00000000000024a8 <_fini>:
    24a8:	f3 0f 1e fa          	endbr64 
    24ac:	48 83 ec 08          	sub    $0x8,%rsp
    24b0:	48 83 c4 08          	add    $0x8,%rsp
    24b4:	c3                   	retq   
