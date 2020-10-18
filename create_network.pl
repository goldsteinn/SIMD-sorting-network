 
use Algorithm::Networksort;

my $num = $ARGV[0] + 0;
my $algo = $ARGV[1];
my $nw = Algorithm::Networksort->new(inputs => $num, algorithm => $algo);
my @cmp = $nw->comparators();
    
print $nw->depth();
#print $nw ,"\n\n";
#foreach my $l1 ( @{ $cmp[0] }) {
#  print join(", ", @{$l1}), ", " ;
#}
#print "\n";



