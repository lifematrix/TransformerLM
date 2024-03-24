
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sstream>
#include <map>


#define MAX_LINE_LEN 1024*1024

int corpus_stats(const char* fileName, int first_n=-1)
{
    int nlines = 0;
    char buffer[MAX_LINE_LEN];
    char word[MAX_LINE_LEN];
    std::map<std::string, int> wordCount;

    FILE *file = fopen(fileName, "r");
    if (file == NULL) {
        perror("Failed to open file");
        return -1;
    }

    char *p, *pw;
    while(1) {
        nlines++;
        if (first_n >= 0 && nlines > first_n)
            break;
        if (fgets(buffer, sizeof(buffer), file) == NULL)
            break;

        p = &buffer[0];
        while(1) {
            if (*p == '\0' || *p=='\n')
                break;
            while(*p == ' ')
                p++;
            pw = &word[0];
            *pw = '\0';
            while(*p != ' ' && *p != '\0' && *p != '\n')
                *(pw++) = *(p++);
            *pw = '\0';
            if (word[0] != '\0') {
                std::string sw(&word[0]);
                wordCount[sw]++;
            }
        }
        if (nlines % 10000 == 0) {
            fprintf(stderr, "process %d\n", nlines);
        }
    }
    fclose(file);

    // Print the word counts
    for (const auto &pair : wordCount) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }
}

int main()
{
    corpus_stats("../../../data/nlp/WMT-14_en-de/train.en", -1);
}