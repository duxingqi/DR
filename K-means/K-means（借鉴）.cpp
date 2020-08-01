#include <stdio.h>
#include <math.h>
#include <malloc.h>
void sort(int n,int d,int c,int**sample,float**centre,int**category);//��������������з���
void ncentre(int n,int d,int c,int**sample,float**centre,int**category);//�����µľ�������
void rcentre(int d,int c,float**ecentre,float**centre);//����һ�ε����Ľ��м�¼���б��Ƿ����� 
int judge(int d,int c,float**ecentre,float**centre);//�б��Ƿ����ѭ��������
int main(){
	int n=0;
	int d=0;
	int c=0;//nΪ�����������dΪ�������ڿռ��ά����cΪҪ��Ϊ�Ķ����� 
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
		sample[i]=(int*)malloc(d*sizeof(int));//�Դ洢�������������洢�ռ�
	float **centre=(float**)malloc(c*sizeof(float*));
	for(i=0;i<c;i++)
	    centre[i]=(float*)malloc(d*sizeof(float));//Ϊ�������ķ���洢�ռ� 
	float **ecentre=(float**)malloc(c*sizeof(float*));
	for(i=0;i<c;i++)
	    ecentre[i]=(float*)malloc(d*sizeof(float));//Ϊ��һ�εľ������ķ���洢�ռ� 
	int **category=(int**)malloc(c*sizeof(int*));
	for(i=0;i<c;i++)
	    category[i]=(int*)malloc(n*sizeof(int));//Ϊ������������ļ�¼����洢�ռ�
	for(i=0;i<n;i++){
 	  printf("the %dth number\n",i);
	  for(int j=0;j<d;j++)
	  		scanf("%d",&sample[i][j]);			
    } //�������������������
    for(i=0;i<c;i++)
    	for(j=0;j<d;j++)
    	   centre[i][j]=sample[i][j];//ѡȡǰC��������Ϊ��ʼ�������Ŀ�ʼ���� 
	do{
    	sort(n,d,c,sample,centre,category);
    	rcentre(d,c,ecentre,centre);
    	ncentre(n,d,c,sample,centre,category);
	}while(judge(d,c,ecentre,centre));/*�˶�Ϊ�������̣���ѭ��1.����2.��¼��һ������3.�����µľ�������
	                                    ��˳�������ֱ����������������һ�㲻�ڱ仯������ѭ�����õ��ֺõ�K�����Լ�K���������ġ�*/ 
	for(i=0;i<c;i++){
	printf("the %dth category is consist of:",i);
       for(j=0;j<n;j++)
        		if(category[i][j]!=-1)
				    printf(" %d  ",category[i][j]);
					printf("\n");
	}//���������˳���0��ʼ��������ţ��˶����ÿһ�������ﺬ����Щ������
	for(i=0;i<c;i++){
	printf("the %dth category center is:",i);
	for(j=0;j<d;j++)
				    printf(" %f  ",ecentre[i][j]);
					printf("\n");
}//�˶��������ÿһ����������� 
	return 0;
}
void sort(int n,int d,int c,int**sample,float**centre,int**category){
    int i=0;
 	int j=0;
 	int k=0; 
 	int t=0;
 	int cc=-1;//����ѭ������ʱ��¼���ڵľ��� 
	float min=-1; 
	float result=0;
	for(i=0;i<c;i++)
        for(j=0;j<n;j++)
        	category[i][j]=-1;//��category��ʼ��
	for(i=0;i<n;i++){
    	for(j=0;j<c;j++){ 
				for(k=0;k<d;k++){
					result+=(sample[i][k]-centre[j][k])*(sample[i][k]-centre[j][k]);//������ŷ�Ͼ����ƽ����Ϊ�жϵı�׼ 
						
				}//�����������J�ľ��� 
				if(min<0||min>result){
					min=result;
					cc=j;
				} //���бȽϲ���¼������̵ľ��� 
				result=0; 
		}//ÿ������ֱ����һ�� 
		while(category[cc][t]!=-1)
			t++;
		category[cc][t]=i;
		t=0;
		min=-1; 
    }//ѭ��ÿ���һ�ν�һ����������뵽һ�������� 
}//����������˶������������һ�η��� 
void rcentre(int d,int c,float**ecentre,float**centre){
    int i=0;
    int j=0;
	for(i=0;i<c;i++)
	     for(j=0;j<d;j++)
	     ecentre[i][j]=centre[i][j];
}//��ecentre����һ�εľ������Ľ��м�¼����������б����������Ƿ��Ѿ�����
void ncentre(int n,int d,int c,int**sample,float**centre,int**category){
    int i=0;
    int j=0;
    int k=0;
    int t=0;
    int* sum=(int*)malloc(d*sizeof(int));//���ڽ�ͬһ�����в�ͬά�ȵ�����ֱ���� 
    for(t=0;t<d;t++)
     	sum[t]=0;
     	for(i=0;i<c;i++){
 	j=0;
 	while(category[i][j]>=0)//ֱ��������û��Ԫ��ֹͣ���� 
 	{
 	 	for(k=0;k<d;k++)
 	 		sum[k]+=sample[category[i][j]][k];//ͬһ����������������֮�� 
 	 	j++; 	 	
 	}
 	for(t=0;t<d;t++)
 		centre[i][t]=(float)sum[t]/j;//��i�������е�j��Ԫ�أ�ÿһ��ά�������ֵ����������µľ������� 
   for(t=0;t<d;t++)
   		sum[t]=0;		//���³�ʼ�� 
 } 
} //��������������·����ľ�������� 
int judge(int d,int c,float**ecentre,float**centre){
	int i=0;
	int j=0;
	for(i=0;i<c;i++)
	     for(j=0;j<d;j++)
	      if(ecentre[i][j]!=centre[i][j])
	      return 1;
	return 0;
}//�������Լ����ľ�����������һ�εľ������Ľ��бȽϣ���ȫ��û�б仯��˵�����������Ѿ��������ڱ仯 


