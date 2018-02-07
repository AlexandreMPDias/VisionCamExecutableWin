#include <stdio.h>
#include <stdlib.h>

#define SIZE 500

int i = 0;
char pre[SIZE][SIZE];
FILE* arq;

char temp[SIZE];

void convert(char own[], char str[]){
    i++;
    fprintf(arq, "(\'gif\\\\%s\\\\%s\',\'.\\\\gif\\\\%s\'),", own, str, own);
    if(i%10 == 0){
        fprintf(arq, "\n");
    }
}

void angry(){
    convert("angry","20171125-174407.mp4");
    convert("angry","20171125-175000.mp4");
    convert("angry","20171125-175332.mp4");
    convert("angry","20171125-175426.mp4");
    convert("angry","20171125-175500.mp4");
    convert("angry","20171125-213610.mp4");
    convert("angry","20171125-215225.mp4");
    convert("angry","20171125-215419.mp4");
    convert("angry","20171125-215442.mp4");
    convert("angry","20171125-215837.mp4");
    convert("angry","20171125-220830.mp4");
    convert("angry","20171125-221426.mp4");
    convert("angry","20171125-222319.mp4");
    convert("angry","20171125-222615.mp4");
    convert("angry","20171125-224106.mp4");
    convert("angry","20171125-224256.mp4");
    convert("angry","20171125-225949.mp4");
    convert("angry","20171125-230223.mp4");
    convert("angry","20171125-231452.mp4");
    convert("angry","20171125-231640.mp4");
    convert("angry","20171125-234253.mp4");
    convert("angry","20171125-235653.mp4");
    convert("angry","20171125-235659.mp4");
    convert("angry","20171126-000355.mp4");
    convert("angry","20171126-000407.mp4");
    convert("angry","20171126-000414.mp4");
    convert("angry","20171126-000432.mp4");
    convert("angry","20171126-001141.mp4");
    convert("angry","20171126-001221.mp4");
    convert("angry","20171126-001715.mp4");
    convert("angry","20171126-001755.mp4");
    convert("angry","20171126-001904.mp4");
    convert("angry","20171126-001936.mp4");
    convert("angry","20171126-002058.mp4");
    convert("angry","20171126-002125.mp4");
    convert("angry","20171126-002252.mp4");
    convert("angry","20171126-002357.mp4");
    convert("angry","20171126-002455.mp4");
    convert("angry","20171126-002508.mp4");
    convert("angry","20171126-002521.mp4");
    convert("angry","20171126-002621.mp4");
    convert("angry","20171126-002646.mp4");
    convert("angry","20171126-002759.mp4");
    convert("angry","20171126-002820.mp4");
    convert("angry","20171126-003232.mp4");
    convert("angry","20171126-003256.mp4");
    convert("angry","20171126-003357.mp4");
    convert("angry","20171126-003510.mp4");
    convert("angry","20171126-003618.mp4");
    convert("angry","20171126-003649.mp4");
    convert("angry","20171126-003720.mp4");
    convert("angry","20171126-003923.mp4");
    convert("angry","20171126-004014.mp4");
    convert("angry","20171126-004114.mp4");
    convert("angry","20171126-004126.mp4");
    convert("angry","20171126-004144.mp4");
    convert("angry","20171126-160149.mp4");
    convert("angry","20171126-160314.mp4");
}

void happy(){
    convert("happy","20171124-141425.mp4");
    convert("happy","20171124-141511.mp4");
    convert("happy","20171124-141736.mp4");
    convert("happy","20171124-141817.mp4");
    convert("happy","20171124-142218.mp4");
    convert("happy","20171125-173121.mp4");
    convert("happy","20171125-173354.mp4");
    convert("happy","20171125-212611.mp4");
    convert("happy","20171125-213333.mp4");
    convert("happy","20171125-213535.mp4");
    convert("happy","20171125-215100.mp4");
    convert("happy","20171125-215327.mp4");
    convert("happy","20171125-215505.mp4");
    convert("happy","20171125-215753.mp4");
    convert("happy","20171125-215907.mp4");
    convert("happy","20171125-220203.mp4");
    convert("happy","20171125-220220.mp4");
    convert("happy","20171125-220408.mp4");
    convert("happy","20171125-220429.mp4");
    convert("happy","20171125-220455.mp4");
    convert("happy","20171125-220704.mp4");
    convert("happy","20171125-220714.mp4");
    convert("happy","20171125-221210.mp4");
    convert("happy","20171125-221410.mp4");
    convert("happy","20171125-222109.mp4");
    convert("happy","20171125-222254.mp4");
    convert("happy","20171125-222302.mp4");
    convert("happy","20171125-222432.mp4");
    convert("happy","20171125-222541.mp4");
    convert("happy","20171125-222804.mp4");
    convert("happy","20171125-223812.mp4");
    convert("happy","20171125-224051.mp4");
    convert("happy","20171125-230518.mp4");
    convert("happy","20171125-231250.mp4");
    convert("happy","20171125-231343.mp4");
    convert("happy","20171125-231427.mp4");
    convert("happy","20171125-231711.mp4");
    convert("happy","20171125-232207.mp4");
    convert("happy","20171125-232358.mp4");
    convert("happy","20171125-235258.mp4");
    convert("happy","20171125-235705.mp4");
    convert("happy","20171125-235818.mp4");
    convert("happy","20171126-000425.mp4");
    convert("happy","20171126-000438.mp4");
    convert("happy","20171126-000446.mp4");
    convert("happy","20171126-001047.mp4");
    convert("happy","20171126-001055.mp4");
    convert("happy","20171126-001103.mp4");
    convert("happy","20171126-001132.mp4");
    convert("happy","20171126-001442.mp4");
    convert("happy","20171126-001840.mp4");
    convert("happy","20171126-001951.mp4");
    convert("happy","20171126-002009.mp4");
    convert("happy","20171126-002049.mp4");
    convert("happy","20171126-002140.mp4");
    convert("happy","20171126-002407.mp4");
    convert("happy","20171126-002531.mp4");
    convert("happy","20171126-002610.mp4");
    convert("happy","20171126-002811.mp4");
    convert("happy","20171126-002831.mp4");
    convert("happy","20171126-002927.mp4");
    convert("happy","20171126-003220.mp4");
    convert("happy","20171126-003413.mp4");
    convert("happy","20171126-004002.mp4");
    convert("happy","20171126-004056.mp4");
    convert("happy","20171126-004121.mp4");
    convert("happy","20171126-004328.mp4");
    convert("happy","20171126-004426.mp4");
    convert("happy","20171126-004459.mp4");
}

void surprised(){
    convert("surprised","20171125-214501.mp4");
    convert("surprised","20171125-214512.mp4");
    convert("surprised","20171125-214639.mp4");
    convert("surprised","20171125-215247.mp4");
    convert("surprised","20171125-215625.mp4");
    convert("surprised","20171125-215652.mp4");
    convert("surprised","20171125-215707.mp4");
    convert("surprised","20171125-220402.mp4");
    convert("surprised","20171125-220419.mp4");
    convert("surprised","20171125-222309.mp4");
    convert("surprised","20171125-235305.mp4");
    convert("surprised","20171125-235639.mp4");
    convert("surprised","20171125-235644.mp4");
    convert("surprised","20171125-235721.mp4");
    convert("surprised","20171125-235734.mp4");
    convert("surprised","20171125-235739.mp4");
    convert("surprised","20171125-235809.mp4");
    convert("surprised","20171126-000149.mp4");
    convert("surprised","20171126-000156.mp4");
    convert("surprised","20171126-001542.mp4");
    convert("surprised","20171126-001643.mp4");
    convert("surprised","20171126-001652.mp4");
    convert("surprised","20171126-001701.mp4");
    convert("surprised","20171126-001725.mp4");
    convert("surprised","20171126-001810.mp4");
    convert("surprised","20171126-002030.mp4");
    convert("surprised","20171126-002222.mp4");
    convert("surprised","20171126-002308.mp4");
    convert("surprised","20171126-002346.mp4");
    convert("surprised","20171126-002416.mp4");
    convert("surprised","20171126-002427.mp4");
    convert("surprised","20171126-002439.mp4");
    convert("surprised","20171126-002502.mp4");
    convert("surprised","20171126-002628.mp4");
    convert("surprised","20171126-002657.mp4");
    convert("surprised","20171126-002724.mp4");
    convert("surprised","20171126-002737.mp4");
    convert("surprised","20171126-003308.mp4");
    convert("surprised","20171126-003330.mp4");
    convert("surprised","20171126-003439.mp4");
    convert("surprised","20171126-003639.mp4");
    convert("surprised","20171126-003746.mp4");
    convert("surprised","20171126-003842.mp4");
    convert("surprised","20171126-004022.mp4");
    convert("surprised","20171126-004033.mp4");
    convert("surprised","20171126-004044.mp4");
    convert("surprised","20171126-004358.mp4");
    convert("surprised","20171126-160324.mp4");
    convert("surprised","20171126-160349.mp4");
    convert("surprised","20171126-160438.mp4");
    convert("surprised","20171126-160515.mp4");
}

void neutral(){
    convert("neutral","20171124-141457.mp4");
    convert("neutral","20171124-141537.mp4");
    convert("neutral","20171125-212749.mp4");
    convert("neutral","20171125-212800.mp4");
    convert("neutral","20171125-212809.mp4");
    convert("neutral","20171125-212818.mp4");
    convert("neutral","20171125-212823.mp4");
    convert("neutral","20171125-212830.mp4");
    convert("neutral","20171125-213001.mp4");
    convert("neutral","20171125-213012.mp4");
    convert("neutral","20171125-213038.mp4");
    convert("neutral","20171125-213111.mp4");
    convert("neutral","20171125-213148.mp4");
    convert("neutral","20171125-213239.mp4");
    convert("neutral","20171125-221025.mp4");
    convert("neutral","20171125-223622.mp4");
    convert("neutral","20171125-225618.mp4");
    convert("neutral","20171125-231232.mp4");
    convert("neutral","20171125-232629.mp4");
    convert("neutral","20171126-004228.mp4");
    convert("neutral","20171126-004234.mp4");
    convert("neutral","20171126-004245.mp4");
    convert("neutral","20171126-004257.mp4");
    convert("neutral","20171126-004303.mp4");
    convert("neutral","20171126-004310.mp4");
    convert("neutral","20171126-004319.mp4");
    convert("neutral","20171126-004334.mp4");
    convert("neutral","20171126-004341.mp4");
    convert("neutral","20171126-004410.mp4");
    convert("neutral","20171126-004437.mp4");
    convert("neutral","20171126-004507.mp4");
    convert("neutral","20171126-004519.mp4");
    convert("neutral","20171126-004529.mp4");
    convert("neutral","20171126-004547.mp4");
    convert("neutral","20171126-004601.mp4");
    convert("neutral","20171126-004625.mp4");
    convert("neutral","20171126-004643.mp4");
    convert("neutral","20171126-160302.mp4");
    convert("neutral","20171126-160329.mp4");
    convert("neutral","20171126-160337.mp4");
    convert("neutral","20171126-160357.mp4");
    convert("neutral","20171126-160405.mp4");
    convert("neutral","20171126-160412.mp4");
    convert("neutral","20171126-160417.mp4");
    convert("neutral","20171126-160445.mp4");
    convert("neutral","20171126-160457.mp4");
    convert("neutral","20171126-160523.mp4");
    convert("neutral","20171126-160528.mp4");
}

int main(void){
   arq = fopen("uhu.txt","w");
   angry();
   happy();
   surprised();
   neutral();
}