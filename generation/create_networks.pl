use Algorithm::Networksort;

my $num = $ARGV[0] + 0;
my $algo = $ARGV[1];
my $nw = Algorithm::Networksort->new(inputs => $num, algorithm => $algo);
my @cmp = $nw->comparators();
    

print $nw ,"\n";
foreach my $l1 ( @{ $cmp[0] }) {
  print join(", ", @{$l1}), ", " ;
}
print "\n";

# 0,1,3,2,2,0,3,1,1,0,3,2,5,4,6,7,4,6,5,7,4,5,6,7,0,4,1,5,2,6,3,7,0,2,1,3,0,1,2,3,4,6,5,7,4,5,6,7
# 0,1,3,2,2,0,3,1,1,0,3,2,5,4,6,7,4,6,5,7,4,5,6,7,0,4,1,5,2,6,3,7,0,2,1,3,0,1,2,3,4,6,5,7,4,5,6,7





