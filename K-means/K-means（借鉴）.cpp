#include <stdio.h>
#include <math.h>
#include <malloc.h>
void sort(int n,int d,int c,int**sample,float**centre,int**category);//对所有样本点进行分类
void ncentre(int n,int d,int c,int**sample,float**centre,int**category);//产生新的聚类中心
void rcentre(int d,int c,float**ecentre,float**centre);//对上一次的中心进行记录以判别是否收敛 
int judge(int d,int c,float**ecentre,float**centre);//判别是否结束循环的条件
int main(){
	int n=0;
	int d=0;
	int c=0;//n为样本点个数，d为样本所在空间的维数，c为要分为的多少组 
	int i=0;
	int j=0;
	printf("Please input the number of samples :\n");
	scanf("%d",&n);
	printf("Please input the dimension of the sample :\n");
	scanf("%d",&d);
	printf("Please input the number of clusters :\n");
	scanf("%d",&c);
	printf("please input the sample number :\n");
	int **sample=(int**)malloc(n*sizeof(int*));
	for(i=0;i<n;i++)
		sample[i]=(int*)malloc(d*sizeof(int));//对存储样本的数组分配存储空间
	float **centre=(float**)malloc(c*sizeof(float*));
	for(i=0;i<c;i++)
	    centre[i]=(float*)malloc(d*sizeof(float));//为聚类中心分配存储空间 
	float **ecentre=(float**)malloc(c*sizeof(float*));
	for(i=0;i<c;i++)
	    ecentre[i]=(float*)malloc(d*sizeof(float));//为上一次的聚类中心分配存储空间 
	int **category=(int**)malloc(c*sizeof(int*));
	for(i=0;i<c;i++)
	    category[i]=(int*)malloc(n*sizeof(int));//为样本的所属类的记录分配存储空间
	for(i=0;i<n;i++){
 	  printf("the %dth number\n",i);
	  for(int j=0;j<d;j++)
	  		scanf("%d",&sample[i][j]);			
    } //输入所有样本点的坐标
    for(i=0;i<c;i++)
    	for(j=0;j<d;j++)
    	   centre[i][j]=sample[i][j];//选取前C个样本点为初始聚类中心开始迭代 
	do{
    	sort(n,d,c,sample,centre,category);
    	rcentre(d,c,ecentre,centre);
    	ncentre(n,d,c,sample,centre,category);
	}while(judge(d,c,ecentre,centre));/*此段为迭代过程，遵循着1.分类2.记录上一次中心3.产生新的聚类中心
	                                    的顺序迭代，直至聚类中心收敛至一点不在变化，结束循环，得到分好的K个类以及K个聚类中心。*/ 
	for(i=0;i<c;i++){
	printf("the %dth category is consist of:",i);
       for(j=0;j<n;j++)
        		if(category[i][j]!=-1)
				    printf(" %d  ",category[i][j]);
					printf("\n");
	}//按照输入的顺序从0开始对样本标号，此段输出每一个聚类里含有哪些样本点
	for(i=0;i<c;i++){
	printf("the %dth category center is:",i);
	for(j=0;j<d;j++)
				    printf(" %f  ",ecentre[i][j]);
					printf("\n");
}//此段输出的是每一个聚类的中心 
	return 0;
}
void sort(int n,int d,int c,int**sample,float**centre,int**category){
    int i=0;
 	int j=0;
 	int k=0; 
 	int t=0;
 	int cc=-1;//用于循环结束时记录属于的聚类 
	float min=-1; 
	float result=0;
	for(i=0;i<c;i++)
        for(j=0;j<n;j++)
        	category[i][j]=-1;//将category初始化
	for(i=0;i<n;i++){
    	for(j=0;j<c;j++){ 
				for(k=0;k<d;k++){
					result+=(sample[i][k]-centre[j][k])*(sample[i][k]-centre[j][k]);//采用了欧氏距离的平方作为判断的标准 
						
				}//计算出到聚类J的距离 
				if(min<0||min>result){
					min=result;
					cc=j;
				} //进行比较并记录距离最短的聚类 
				result=0; 
		}//每个聚类分别计算一次 
		while(category[cc][t]!=-1)
			t++;
		category[cc][t]=i;
		t=0;
		min=-1; 
    }//循环每完成一次将一个样本点加入到一个聚类里 
}//本函数完成了对所有样本点的一次分类 
void rcentre(int d,int c,float**ecentre,float**centre){
    int i=0;
    int j=0;
	for(i=0;i<c;i++)
	     for(j=0;j<d;j++)
	     ecentre[i][j]=centre[i][j];
}//用ecentre对上一次的聚类中心进行记录，方便后面判别聚类的中心是否已经收敛
void ncentre(int n,int d,int c,int**sample,float**centre,int**category){
    int i=0;
    int j=0;
    int k=0;
    int t=0;
    int* sum=(int*)malloc(d*sizeof(int));//用于将同一聚类中不同维度的坐标分别求和 
    for(t=0;t<d;t++)
     	sum[t]=0;
     	for(i=0;i<c;i++){
 	j=0;
 	while(category[i][j]>=0)//直到聚类中没有元素停止遍历 
 	{
 	 	for(k=0;k<d;k++)
 	 		sum[k]+=sample[category[i][j]][k];//同一聚类的样本坐标相加之和 
 	 	j++; 	 	
 	}
 	for(t=0;t<d;t++)
 		centre[i][t]=(float)sum[t]/j;//第i个聚类中的j个元素，每一个维度上求均值，即可求得新的聚类中心 
   for(t=0;t<d;t++)
   		sum[t]=0;		//重新初始化 
 } 
} //本函数计算出重新分类后的聚类的中心 
int judge(int d,int c,float**ecentre,float**centre){
	int i=0;
	int j=0;
	for(i=0;i<c;i++)
	     for(j=0;j<d;j++)
	      if(ecentre[i][j]!=centre[i][j])
	      return 1;
	return 0;
}//本函数对计算后的聚类中心与上一次的聚类中心进行比较，若全都没有变化则说明聚类中心已经收敛不在变化 


