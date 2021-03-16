from django.shortcuts import render


def board_view(request):
    return render(request, 'board/index.html')
